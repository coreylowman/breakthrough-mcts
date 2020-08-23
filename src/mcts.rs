use crate::env::{Env, BLACK, WHITE};
use crate::rand::rngs::StdRng;
use crate::rand::SeedableRng;
use std::collections::VecDeque;
use std::time::Instant;

pub struct Node<E: Env + Clone> {
    pub id: usize,
    pub parent: usize,
    pub env: E,
    pub terminal: bool,
    pub expanded: bool,
    pub my_action: bool,
    pub actions: Vec<E::Action>,
    pub children: Vec<usize>,
    pub unvisited_actions: E::ActionIterator,
    pub num_visits: f32,
    pub reward: f32,
}

impl<E: Env + Clone> Node<E> {
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.actions.capacity() * std::mem::size_of::<E::Action>()
            + self.children.capacity() * std::mem::size_of::<usize>()
            // + self.unvisited_actions.capacity() * std::mem::size_of::<E::Action>()
            + std::mem::size_of::<E::ActionIterator>()
    }

    pub fn new_root(id: usize) -> Self {
        let env = E::new();
        let actions = env.iter_actions();
        Node {
            id: id,
            parent: 0,
            env: env,
            terminal: false,
            expanded: false,
            my_action: false,
            actions: Vec::new(),
            children: Vec::new(),
            unvisited_actions: actions,
            num_visits: 0.0,
            reward: 0.0,
        }
    }

    pub fn new(id: usize, node: &Self, action: &E::Action) -> Self {
        let mut env = node.env.clone();
        let is_over = env.step(action);
        let actions = env.iter_actions();
        Node {
            id: id,
            parent: node.id,
            env: env,
            terminal: is_over,
            expanded: is_over,
            my_action: !node.my_action,
            actions: Vec::new(),
            children: Vec::new(),
            unvisited_actions: actions,
            num_visits: 0.0,
            reward: 0.0,
        }
    }
}

pub fn default_node_value<E: Env + Clone>(nodes: &VecDeque<Node<E>>, node: &Node<E>) -> f32 {
    node.reward / node.num_visits
}

pub fn minimax_value<E: Env + Clone>(nodes: &VecDeque<Node<E>>, node: &Node<E>) -> f32 {
    let mut min_value = std::f32::INFINITY;
    let root_id = nodes[0].id;

    for (i, &child_id) in node.children.iter().enumerate() {
        let child = &nodes[child_id - root_id];
        let value = 1.0 - child.reward / child.num_visits;
        if value < min_value {
            min_value = value;
        }
    }

    min_value
}

pub struct MCTS<E: Env + Clone> {
    pub id: bool,
    pub root: usize,
    pub nodes: VecDeque<Node<E>>,
    pub evaluator: fn(&VecDeque<Node<E>>, &Node<E>) -> f32,
    pub rng: StdRng, // note: this is about the same performance as SmallRng or any of the XorShiftRngs that got moved to the xorshift crate
}

impl<E: Env + Clone> MCTS<E> {
    pub fn memory_usage(&self) -> usize {
        let mut mem = std::mem::size_of_val(&self);
        for node in self.nodes.iter() {
            mem += node.memory_usage();
        }
        mem += (self.nodes.capacity() - self.nodes.len()) * std::mem::size_of::<Node<E>>();
        mem
    }

    pub fn new(id: bool, evaluator: fn(&VecDeque<Node<E>>, &Node<E>) -> f32) -> Self {
        let mut root = Node::new_root(0);
        root.my_action = id == WHITE;
        let mut nodes = VecDeque::new();
        nodes.push_back(root);
        Self {
            id: id,
            root: 0,
            nodes: nodes,
            rng: StdRng::seed_from_u64(0),
            evaluator: evaluator,
        }
    }

    pub fn with_capacity(
        id: bool,
        capacity: usize,
        evaluator: fn(&VecDeque<Node<E>>, &Node<E>) -> f32,
        seed: u64,
    ) -> Self {
        let mut root = Node::new_root(0);
        root.my_action = id == WHITE;
        let mut nodes = VecDeque::with_capacity(capacity);
        nodes.push_back(root);
        Self {
            id: id,
            root: 0,
            nodes: nodes,
            rng: StdRng::seed_from_u64(seed),
            evaluator: evaluator,
        }
    }

    fn next_node_id(&self) -> usize {
        self.nodes.len() + self.root
    }

    pub fn step_action(&mut self, action: &E::Action) {
        self.root = match self.nodes[self.root - self.root]
            .actions
            .iter()
            .position(|a| a == action)
        {
            Some(action_index) => {
                let new_root = self.nodes[self.root - self.root].children[action_index];
                drop(self.nodes.drain(0..new_root - self.root));
                new_root
            }
            None => {
                let child_node = Node::new(0, &self.nodes[self.root - self.root], action);
                self.nodes.clear();
                self.nodes.push_back(child_node);
                0
            }
        };

        self.nodes[0].parent = self.root;
    }

    pub fn best_action(&self) -> E::Action {
        let root = &self.nodes[self.root - self.root];

        let mut best_action_ind = 0;
        let mut best_value = -std::f32::INFINITY;

        for (i, &child_id) in root.children.iter().enumerate() {
            let child = &self.nodes[child_id - self.root];
            let value = (self.evaluator)(&self.nodes, child);
            if value > best_value {
                best_value = value;
                best_action_ind = i;
            }
        }

        root.actions[best_action_ind].clone()
    }

    fn explore(&mut self) {
        let node_id = self.select_node();
        let reward = self.rollout(node_id);
        self.backprop(node_id, reward);
    }

    fn select_node(&mut self) -> usize {
        let mut node_id = self.root;
        loop {
            // assert!(node_id < self.nodes.len());
            let node = &self.nodes[node_id - self.root];
            if node.terminal {
                return node_id;
            } else if node.expanded {
                node_id = self.select_best_child(node_id);
            } else {
                return self.select_unexpanded_child(node_id);
            }
        }
    }

    fn select_best_child(&mut self, node_id: usize) -> usize {
        // assert!(node_id < self.nodes.len());

        let node = &self.nodes[node_id - self.root];

        let mut best_child = 0;
        let mut best_value = -std::f32::INFINITY;

        let parent_visits = node.num_visits.log(2.0);

        for &child_id in node.children.iter() {
            let child = &self.nodes[child_id - self.root];

            let value =
                child.reward / child.num_visits + (2.0 * parent_visits / child.num_visits).sqrt();

            if value > best_value {
                best_value = value;
                best_child = child_id;
            }
        }

        best_child
    }

    // #[inline(never)]
    fn select_unexpanded_child(&mut self, node_id: usize) -> usize {
        let child_id = self.next_node_id();

        let child_node = {
            let node = &mut self.nodes[node_id - self.root];
            let action = node.unvisited_actions.next().unwrap();
            if node.actions.capacity() == 0 {
                // finally allocate space for max possible actions if we are looking down this node
                node.actions.reserve_exact(48);
                node.children.reserve_exact(48);
            }
            node.actions.push(action);
            node.children.push(child_id);
            if node.unvisited_actions.size_hint().0 == 0 {
                node.expanded = true;
                // note: very little impact to memory
                // node.unvisited_actions.shrink_to_fit();
                // node.actions.shrink_to_fit();
                // node.children.shrink_to_fit();
            }
            Node::new(child_id, &node, &action)
        };
        self.nodes.push_back(child_node);

        child_id
    }

    fn rollout(&mut self, node_id: usize) -> f32 {
        // assert!(node_id < self.nodes.len());
        let mut env = self.nodes[node_id - self.root].env.clone();
        let mut is_over = env.is_over();
        while !is_over {
            let action = env.get_random_action(&mut self.rng);
            is_over = env.step(&action);
        }
        env.reward(self.id)
    }

    fn backprop(&mut self, leaf_node_id: usize, reward: f32) {
        let mut node_id = leaf_node_id;
        loop {
            // assert!(node_id < self.nodes.len());

            let node = &mut self.nodes[node_id - self.root];

            node.num_visits += 1.0;

            // note this is reversed because its actually the previous node's action that this node's reward is associated with
            node.reward += if !node.my_action {
                reward
            } else {
                1.0 - reward
            };

            if node_id == self.root {
                break;
            }

            node_id = node.parent;
        }
    }

    pub fn explore_for(&mut self, millis: u128) -> (usize, u128) {
        let start = Instant::now();
        let mut steps = 0;
        while start.elapsed().as_millis() < millis {
            self.explore();
            steps += 1;
        }
        (steps, start.elapsed().as_millis())
    }

    pub fn explore_n(&mut self, n: usize) -> (usize, u128) {
        let start = Instant::now();
        for _i in 0..n {
            self.explore();
        }
        (n, start.elapsed().as_millis())
    }

    pub fn timed_explore_n(&mut self, n: usize) -> (usize, u128) {
        let mut select_ns = 0;
        let mut select_best_ns = 0;
        let mut select_unexpanded_ns = 0;
        let mut rollout_ns = 0;
        let mut backprop_ns = 0;
        let mut select_best_n = 0;
        let mut select_unexpanded_n = 0;

        let start = Instant::now();
        for _i in 0..n {
            let select_start = Instant::now();
            // let node_id = self.select_node();
            let node_id = {
                let mut node_id = self.root;
                loop {
                    // assert!(node_id < self.nodes.len());
                    let node = &self.nodes[node_id - self.root];
                    if node.terminal {
                        break;
                    } else if node.expanded {
                        select_best_n += 1;
                        let select_best_start = Instant::now();
                        node_id = self.select_best_child(node_id);
                        select_best_ns += select_best_start.elapsed().as_nanos();
                    } else {
                        select_unexpanded_n += 1;
                        let select_unexpanded_start = Instant::now();
                        node_id = self.select_unexpanded_child(node_id);
                        select_unexpanded_ns += select_unexpanded_start.elapsed().as_nanos();
                        break;
                    }
                }
                node_id
            };
            select_ns += select_start.elapsed().as_nanos();

            let rollout_start = Instant::now();
            let reward = self.rollout(node_id);
            rollout_ns += rollout_start.elapsed().as_nanos();

            let backprop_start = Instant::now();
            self.backprop(node_id, reward);
            backprop_ns += backprop_start.elapsed().as_nanos();
        }

        println!(
            "select {}ns | rollout {}ns | backprop {}ns",
            select_ns as f32 / n as f32,
            rollout_ns as f32 / n as f32,
            backprop_ns as f32 / n as f32
        );
        println!(
            "select_best_child {}ns | select_unexpanded_child {}ns",
            select_best_ns as f32 / select_best_n as f32,
            select_unexpanded_ns as f32 / select_unexpanded_n as f32
        );
        (n, start.elapsed().as_millis())
    }
}
