use crate::env::{Env, BLACK, WHITE};
use crate::rand::rngs::StdRng;
use crate::rand::SeedableRng;
use std::time::Instant;

pub struct Node<E: Env + Clone> {
    pub parent: usize,                     // 8 bytes
    pub env: E,                            // 32 bytes
    pub terminal: bool,                    // 1 byte
    pub expanded: bool,                    // 1 byte
    pub my_action: bool,                   // 1 byte
    pub children: Vec<(E::Action, usize)>, // 24 bytes
    pub reward: f32,                       // 4 bytes
    pub num_visits: f32,                   // 4 bytes
}

impl<E: Env + Clone> Node<E> {
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            // + self.actions.capacity() * std::mem::size_of::<E::Action>()
            + self.children.capacity() * std::mem::size_of::<(E::Action, usize)>()
            // + self.unvisited_actions.capacity() * std::mem::size_of::<E::Action>()
            + std::mem::size_of::<E::ActionIterator>()
    }

    pub fn new_root(my_action: bool) -> Self {
        Node {
            parent: 0,
            env: E::new(),
            terminal: false,
            expanded: false,
            my_action: my_action,
            children: Vec::new(),
            num_visits: 0.0,
            reward: 0.0,
        }
    }

    pub fn new(parent_id: usize, node: &Self, action: &E::Action) -> Self {
        let mut env = node.env.clone();
        let is_over = env.step(action);
        Node {
            parent: parent_id,
            env: env,
            terminal: is_over,
            expanded: is_over,
            my_action: !node.my_action,
            children: Vec::new(),
            num_visits: 0.0,
            reward: 0.0,
        }
    }
}

pub struct MCTS<E: Env + Clone> {
    pub id: bool,
    pub root: usize,
    pub nodes: Vec<Node<E>>,
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

    pub fn with_capacity(id: bool, capacity: usize, seed: u64) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        let root = Node::new_root(id == WHITE);
        nodes.push(root);
        Self {
            id: id,
            root: 0,
            nodes: nodes,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn next_node_id(&self) -> usize {
        self.nodes.len() + self.root
    }

    pub fn step_action(&mut self, action: &E::Action) {
        self.root = match self.nodes[self.root - self.root]
            .children
            .iter()
            .position(|(a, c)| a == action)
        {
            Some(action_index) => {
                let (_, new_root) = self.nodes[self.root - self.root].children[action_index];
                drop(self.nodes.drain(0..new_root - self.root));
                new_root
            }
            None => {
                let child_node = Node::new(0, &self.nodes[self.root - self.root], action);
                self.nodes.clear();
                self.nodes.push(child_node);
                0
            }
        };

        self.nodes[0].parent = self.root;
    }

    pub fn best_action(&self) -> E::Action {
        // TODO replace this with depth limited alpha beta pruning
        let root = &self.nodes[self.root - self.root];

        let mut best_action_ind = 0;
        let mut best_value = -std::f32::INFINITY;

        for (i, &(_, child_id)) in root.children.iter().enumerate() {
            let child = &self.nodes[child_id - self.root];
            let value = child.reward / child.num_visits;
            if value > best_value {
                best_value = value;
                best_action_ind = i;
            }
        }

        root.children[best_action_ind].0.clone()
    }

    pub fn negamax(&self, depth: u8) -> E::Action {
        self.negamax_search(self.root, depth, -1.0).0.unwrap()
    }

    fn negamax_search(&self, node_id: usize, depth: u8, color: f32) -> (Option<E::Action>, f32) {
        let node = &self.nodes[node_id - self.root];
        if depth == 0 || node.terminal || !node.expanded {
            return (None, color * node.reward / node.num_visits);
        }

        let mut best_value = -std::f32::INFINITY;
        let mut best_action_ind = 0;
        for (i, &(_, child_id)) in node.children.iter().enumerate() {
            let (_, v) = self.negamax_search(child_id, depth - 1, -color);
            if -v > best_value {
                best_value = -v;
                best_action_ind = i;
            }
        }
        (Some(node.children[best_action_ind].0), best_value)
    }

    fn explore(&mut self) {
        let mut node_id = self.root;
        loop {
            // assert!(node_id < self.nodes.len());
            let node = &mut self.nodes[node_id - self.root];
            if node.terminal {
                let reward = node.env.reward(self.id);
                self.backprop(node_id, reward, 1.0);
                return;
            } else if node.expanded {
                node_id = self.select_best_child(node_id);
            } else {
                // reserve max number of actions for children to reduce allocations
                node.children.reserve_exact(48);

                // we are adding all children at once, so this node is about to be expanded
                node.expanded = true;

                let mut total_reward = 0.0;
                let mut total_visits = 0.0;

                // iterate through all the children!
                // TODO take advantage of any symmetry that exist in the game
                for action in node.env.iter_actions() {
                    // TODO calculate this as offset from beginning of expansion?
                    let child_id = self.next_node_id();

                    // create the child node and sample a reward from it
                    let (child_node, reward) = {
                        let node = &mut self.nodes[node_id - self.root];
                        node.children.push((action, child_id));

                        // create the child node... note we will be modifying num_visits and reward later, so mutable
                        let mut child_node = Node::new(node_id, &node, &action);

                        // rollout child to get initial reward
                        let reward = self.rollout(child_node.env.clone());

                        // store initial reward & 1 visit
                        child_node.num_visits = 1.0;
                        child_node.reward = reward;

                        (child_node, reward)
                    };

                    self.nodes.push(child_node);

                    // keep track of reward here so we can backprop 1 time for all the new children
                    total_reward += reward;
                    total_visits += 1.0;
                }

                // backprop all new children rewards back up
                self.backprop(node_id, total_reward, total_visits);

                // we've expanded one node now, 1 round of exploring done!
                return;
            }
        }
    }

    // fn select_node(&mut self) -> usize {
    //     let mut node_id = self.root;
    //     loop {
    //         // assert!(node_id < self.nodes.len());
    //         let node = &self.nodes[node_id - self.root];
    //         if node.terminal {
    //             // TODO check if a double fetch of the node happens from this
    //             return node_id;
    //         } else if node.expanded {
    //             node_id = self.select_best_child(node_id);
    //         } else {
    //             return self.select_unvisited_child(node_id);
    //         }
    //     }
    // }

    // #[inline(never)]
    fn select_best_child(&mut self, node_id: usize) -> usize {
        // assert!(node_id < self.nodes.len());

        let node = &self.nodes[node_id - self.root];

        let mut best_child = 0;
        let mut best_value = -std::f32::INFINITY;

        let visits = node.num_visits.log(2.0);

        // TODO vectorize this since we know all children are next to each other
        for &(_, child_id) in node.children.iter() {
            let child = &self.nodes[child_id - self.root];

            let value = child.reward / child.num_visits + (2.0 * visits / child.num_visits).sqrt();

            if value > best_value {
                best_value = value;
                best_child = child_id;
            }
        }

        best_child
    }

    // #[inline(never)]
    // fn select_unvisited_child(&mut self, parent_id: usize) -> usize {
    //     let child_id = self.next_node_id();

    //     let child_node = {
    //         let node = &mut self.nodes[parent_id - self.root];
    //         let action = node.unvisited_actions.next().unwrap();
    //         if node.children.capacity() == 0 {
    //             // finally allocate space for max possible actions if we are looking down this node
    //             node.children.reserve_exact(48);
    //         }
    //         node.children.push((action, child_id));
    //         if node.unvisited_actions.size_hint().0 == 0 {
    //             node.expanded = true;
    //             // note: very little impact to memory
    //             // node.unvisited_actions.shrink_to_fit();
    //             // node.actions.shrink_to_fit();
    //             // node.children.shrink_to_fit();
    //         }
    //         Node::new(parent_id, &node, &action)
    //     };
    //     self.nodes.push_back(child_node);

    //     child_id
    // }

    fn rollout(&mut self, mut env: E) -> f32 {
        // assert!(node_id < self.nodes.len());
        // note: checking if env.is_over() before cloning doesn't make much difference
        let mut is_over = env.is_over();
        while !is_over {
            let action = env.get_random_action(&mut self.rng);
            is_over = env.step(&action);
        }
        env.reward(self.id)
    }

    fn backprop(&mut self, leaf_node_id: usize, reward: f32, num_visits: f32) {
        let mut node_id = leaf_node_id;
        loop {
            // assert!(node_id < self.nodes.len());

            let node = &mut self.nodes[node_id - self.root];

            node.num_visits += num_visits;

            // TODO multiply reward by -1 instead of this if every time
            // note this is reversed because its actually the previous node's action that this node's reward is associated with
            node.reward += if !node.my_action { reward } else { -reward };

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
        let start_n = self.nodes.len();
        let target_n = start_n + n;
        for _ in 0..n {
            self.explore();
            if self.nodes.len() >= target_n {
                break;
            }
        }
        (self.nodes.len() - start_n, start.elapsed().as_millis())
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
        // for _i in 0..n {
        //     let select_start = Instant::now();
        //     // let node_id = self.select_node();
        //     let node_id = {
        //         let mut node_id = self.root;
        //         loop {
        //             // assert!(node_id < self.nodes.len());
        //             let node = &self.nodes[node_id - self.root];
        //             if node.terminal {
        //                 break;
        //             } else if node.expanded {
        //                 select_best_n += 1;
        //                 let select_best_start = Instant::now();
        //                 node_id = self.select_best_child(node_id);
        //                 select_best_ns += select_best_start.elapsed().as_nanos();
        //             } else {
        //                 select_unexpanded_n += 1;
        //                 let select_unexpanded_start = Instant::now();
        //                 node_id = self.select_unvisited_child(node_id);
        //                 select_unexpanded_ns += select_unexpanded_start.elapsed().as_nanos();
        //                 break;
        //             }
        //         }
        //         node_id
        //     };
        //     select_ns += select_start.elapsed().as_nanos();

        //     let rollout_start = Instant::now();
        //     let reward = self.rollout(node_id);
        //     rollout_ns += rollout_start.elapsed().as_nanos();

        //     let backprop_start = Instant::now();
        //     self.backprop(node_id, reward);
        //     backprop_ns += backprop_start.elapsed().as_nanos();
        // }

        println!(
            "select {}ns | rollout {}ns | backprop {}ns",
            select_ns as f32 / n as f32,
            rollout_ns as f32 / n as f32,
            backprop_ns as f32 / n as f32
        );
        println!(
            "select_best_child {}ns ({}) | select_unexpanded_child {}ns ({})",
            select_best_ns as f32 / select_best_n as f32,
            select_best_n,
            select_unexpanded_ns as f32 / select_unexpanded_n as f32,
            select_unexpanded_n,
        );
        (n, start.elapsed().as_millis())
    }
}
