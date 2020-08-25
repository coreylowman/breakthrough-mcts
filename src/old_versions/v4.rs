extern crate rand;

use crate::rand::rngs::StdRng;
use crate::rand::Rng;
use crate::rand::SeedableRng;

use std::collections::VecDeque;
use std::io;
use std::time::Instant;

macro_rules! parse_input {
    ($x:expr, $t:ident) => {
        $x.trim().parse::<$t>().unwrap()
    };
}

pub const WHITE: bool = true;
pub const BLACK: bool = false;

pub trait Env {
    type Action: Eq + Clone + Copy + std::fmt::Debug;
    type ActionIterator: Iterator<Item = Self::Action>;

    fn new() -> Self;
    fn turn(&self) -> bool;
    fn is_over(&self) -> bool;
    fn reward(&self, color: bool) -> f32;
    fn iter_actions(&self) -> Self::ActionIterator;
    fn actions(&self) -> Vec<Self::Action>;
    fn get_random_action(&self, rng: &mut StdRng) -> Self::Action;
    fn step(&mut self, action: &Self::Action) -> bool;
}

type BitBoard = u64;
type Square = u32;

const COL_1: BitBoard = 0x0101010101010101u64;
const NOT_COL_1: BitBoard = !COL_1;
const COL_8: BitBoard = COL_1 << 7;
const NOT_COL_8: BitBoard = !COL_8;
const ROW_1: BitBoard = 0xFFu64;
const ROW_2: BitBoard = ROW_1 << 8;
const ROW_7: BitBoard = ROW_1 << 48;
const ROW_8: BitBoard = ROW_1 << 56;
// const FULL_BOARD: BitBoard = 0xFFFFFFFFFFFFFFFFu64;

pub struct ActionIterator(BitBoard, BitBoard);

// TODO how is this slower???
// impl ActionIterator {
//     #[inline]
//     fn fast_nth(&mut self, mut n: u32) -> (Square, Square) {
//         while n != 0 {
//             self.0 &= self.0.wrapping_sub(1);
//             self.1 &= self.1.wrapping_sub(1);
//             n -= 1;
//         }
//         let from_sq = self.0.trailing_zeros();
//         let to_sq = self.1.trailing_zeros();
//         (from_sq, to_sq)
//     }
// }

impl Iterator for ActionIterator {
    type Item = (Square, Square);

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }

        let from_sq = self.0.trailing_zeros();
        let to_sq = self.1.trailing_zeros();

        self.0 &= self.0.wrapping_sub(1);
        self.1 &= self.1.wrapping_sub(1);

        Some((from_sq, to_sq))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0 as usize, None)
    }
}

#[derive(Eq, PartialEq, Clone)]
pub struct PlayerInfo {
    id: bool,
    won: bool,
    fwd_shift: u8,
    right_shift: u8,
    left_shift: u8,
    ty_shift: u8,
}

#[derive(Eq, PartialEq, Clone)]
pub struct BitBoardEnv {
    my_bb: BitBoard,
    op_bb: BitBoard,
    me: PlayerInfo,
    op: PlayerInfo,
}

impl BitBoardEnv {
    fn action_bitboards(&self) -> (BitBoard, BitBoard, BitBoard) {
        let fwd_to = self.my_bb.rotate_left(self.me.fwd_shift as u32) & !self.my_bb & !self.op_bb;
        let right_to =
            (self.my_bb & NOT_COL_8).rotate_left(self.me.right_shift as u32) & !self.my_bb;
        let left_to = (self.my_bb & NOT_COL_1).rotate_left(self.me.left_shift as u32) & !self.my_bb;

        let fwd_win = fwd_to & (ROW_1 << self.me.ty_shift);
        let right_win = right_to & (ROW_1 << self.me.ty_shift);
        let left_win = left_to & (ROW_1 << self.me.ty_shift);

        if fwd_win != 0 || right_win != 0 || left_win != 0 {
            (fwd_win, right_win, left_win)
        } else {
            (fwd_to, right_to, left_to)
        }

        // note: this branchless version is actually slower!
        // let raw_any_win = fwd_win != 0 || right_win != 0 || left_win != 0;
        // let any_win = raw_any_win as u64;
        // let none_win = !raw_any_win as u64;

        // let fwd_to = (fwd_win * any_win) | (fwd_to * none_win);
        // let right_to = (right_win * any_win) | (right_to * none_win);
        // let left_to = (left_win * any_win) | (left_to * none_win);

        // (fwd_to, right_to, left_to)
    }
}

impl Env for BitBoardEnv {
    type Action = (Square, Square);
    type ActionIterator =
        std::iter::Chain<std::iter::Chain<ActionIterator, ActionIterator>, ActionIterator>;
    /*
    {
        a: ActionIterator
        b: {
            a: ActionIterator
            b: ActionIterator
            state: ChainState
        }
        state: ChainState
    }
    */

    fn new() -> BitBoardEnv {
        BitBoardEnv {
            my_bb: ROW_1 | ROW_2,
            op_bb: ROW_7 | ROW_8,
            me: PlayerInfo {
                id: WHITE,
                left_shift: 7,
                fwd_shift: 8,
                right_shift: 9,
                won: false,
                ty_shift: 56,
            },
            op: PlayerInfo {
                id: BLACK,
                left_shift: 55,
                fwd_shift: 56,
                right_shift: 57,
                won: false,
                ty_shift: 0,
            },
        }
    }

    fn turn(&self) -> bool {
        self.me.id
    }

    fn is_over(&self) -> bool {
        self.op.won
    }

    fn reward(&self, color: bool) -> f32 {
        // assert!(self.op.won);
        if self.op.id == color {
            1.0
        } else {
            0.0
        }
    }

    fn actions(&self) -> Vec<Self::Action> {
        let mut acs = Vec::with_capacity(16 * 3);

        let (fwd_to, right_to, left_to) = self.action_bitboards();

        let fwd_from = fwd_to.rotate_right(self.me.fwd_shift as u32);
        let right_from = right_to.rotate_right(self.me.right_shift as u32);
        let left_from = left_to.rotate_right(self.me.left_shift as u32);

        acs.extend(ActionIterator(fwd_from, fwd_to));
        acs.extend(ActionIterator(right_from, right_to));
        acs.extend(ActionIterator(left_from, left_to));
        acs
    }

    fn iter_actions(&self) -> Self::ActionIterator {
        let (fwd_to, right_to, left_to) = self.action_bitboards();
        let fwd_from = fwd_to.rotate_right(self.me.fwd_shift as u32);
        let right_from = right_to.rotate_right(self.me.right_shift as u32);
        let left_from = left_to.rotate_right(self.me.left_shift as u32);
        ActionIterator(fwd_from, fwd_to)
            .chain(ActionIterator(right_from, right_to))
            .chain(ActionIterator(left_from, left_to))
    }

    fn get_random_action(&self, rng: &mut StdRng) -> Self::Action {
        let (fwd_to, right_to, left_to) = self.action_bitboards();

        let num_fwd_acs = fwd_to.count_ones() as u8;
        let num_right_acs = right_to.count_ones() as u8;
        let num_left_acs = left_to.count_ones() as u8;

        let i = rng.gen_range(0, num_fwd_acs + num_right_acs + num_left_acs);

        if i >= num_fwd_acs + num_right_acs {
            // generate a left action
            let left_from = left_to.rotate_right(self.me.left_shift as u32);
            ActionIterator(left_from, left_to)
                .nth((i - (num_fwd_acs + num_right_acs)) as usize)
                .unwrap()
        } else if i >= num_fwd_acs {
            // generate a right action
            let right_from = right_to.rotate_right(self.me.right_shift as u32);
            ActionIterator(right_from, right_to)
                .nth((i - num_fwd_acs) as usize)
                .unwrap()
        } else {
            // generate a forward action
            let fwd_from = fwd_to.rotate_right(self.me.fwd_shift as u32);
            ActionIterator(fwd_from, fwd_to).nth(i as usize).unwrap()
        }
    }

    fn step(&mut self, action: &Self::Action) -> bool {
        let &(from_sq, to_sq) = action;

        // note: doing an xor here instead makes no difference, even if you use the same mask for op_bb & my_bb
        self.op_bb &= !(1 << to_sq);
        self.my_bb = (self.my_bb | (1 << to_sq)) & !(1 << from_sq);

        // note: comparing ty_shift to to_bb is faster than doing a ty_shift <= to_sq < ty_max
        self.me.won = (ROW_1 << self.me.ty_shift) & (1 << to_sq) != 0 || self.op_bb == 0;

        std::mem::swap(&mut self.me, &mut self.op);
        std::mem::swap(&mut self.my_bb, &mut self.op_bb);

        self.op.won
    }
}

pub struct Node<E: Env + Clone> {
    pub parent: usize,                        // 8 bytes
    pub env: E,                               // 32 bytes
    pub terminal: bool,                       // 1 byte
    pub expanded: bool,                       // 1 byte
    pub my_action: bool,                      // 1 byte
    pub children: Vec<(E::Action, usize)>,    // 24 bytes
    pub unvisited_actions: E::ActionIterator, // 64 bytes
    pub num_visits: f32,                      //4 bytes
    pub reward: f32,                          // 4 bytes
}

impl<E: Env + Clone> Node<E> {
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            // + self.actions.capacity() * std::mem::size_of::<E::Action>()
            + self.children.capacity() * std::mem::size_of::<(E::Action, usize)>()
            // + self.unvisited_actions.capacity() * std::mem::size_of::<E::Action>()
            + std::mem::size_of::<E::ActionIterator>()
    }

    pub fn new_root(id: usize) -> Self {
        let env = E::new();
        let actions = env.iter_actions();
        Node {
            parent: 0,
            env: env,
            terminal: false,
            expanded: false,
            my_action: false,
            children: Vec::new(),
            unvisited_actions: actions,
            num_visits: 0.0,
            reward: 0.0,
        }
    }

    pub fn new(parent_id: usize, node: &Self, action: &E::Action) -> Self {
        let mut env = node.env.clone();
        let is_over = env.step(action);
        let actions = env.iter_actions();
        Node {
            parent: parent_id,
            env: env,
            terminal: is_over,
            expanded: is_over,
            my_action: !node.my_action,
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
    // let root_id = nodes[0].id;

    // for (i, &child_id) in node.children.iter().enumerate() {
    //     let child = &nodes[child_id - root_id];
    //     let value = 1.0 - child.reward / child.num_visits;
    //     if value < min_value {
    //         min_value = value;
    //     }
    // }

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
                self.nodes.push_back(child_node);
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
                // TODO check if a double fetch of the node happens from this
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

        for &(_, child_id) in node.children.iter() {
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
    fn select_unexpanded_child(&mut self, parent_id: usize) -> usize {
        let child_id = self.next_node_id();

        let child_node = {
            let node = &mut self.nodes[parent_id - self.root];
            let action = node.unvisited_actions.next().unwrap();
            if node.children.capacity() == 0 {
                // finally allocate space for max possible actions if we are looking down this node
                node.children.reserve_exact(48);
            }
            node.children.push((action, child_id));
            if node.unvisited_actions.size_hint().0 == 0 {
                node.expanded = true;
                // note: very little impact to memory
                // node.unvisited_actions.shrink_to_fit();
                // node.actions.shrink_to_fit();
                // node.children.shrink_to_fit();
            }
            Node::new(parent_id, &node, &action)
        };
        self.nodes.push_back(child_node);

        child_id
    }

    fn rollout(&mut self, node_id: usize) -> f32 {
        // assert!(node_id < self.nodes.len());
        // note: checking if env.is_over() before cloning doesn't make much difference
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

fn parse_move(action_str: &String) -> (u32, u32) {
    let action: Vec<char> = action_str.chars().collect();
    let from_x = action[0] as i32 - 'a' as i32;
    let from_y = action[1] as i32 - '1' as i32;
    let to_x = action[2] as i32 - 'a' as i32;
    let to_y = action[3] as i32 - '1' as i32;
    ((from_y * 8 + from_x) as u32, (to_y * 8 + to_x) as u32)
}

fn serialize_move(action: &(u32, u32)) -> String {
    let (from, to) = *action;
    let from_y = from / 8;
    let from_x = from % 8;
    let to_y = to / 8;
    let to_x = to % 8;
    format!(
        "{}{}{}{}",
        (from_x as u8 + 'a' as u8) as char,
        from_y + 1,
        (to_x as u8 + 'a' as u8) as char,
        to_y + 1
    )
}

fn main() {
    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let opponent_move_string = input_line.trim_matches('\n').to_string(); // last move played or "None"
    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let num_legal_moves = parse_input!(input_line, i32); // number of legal moves
    let mut legal_moves = Vec::with_capacity(num_legal_moves as usize);
    for _i in 0..num_legal_moves as usize {
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let move_string = input_line.trim().to_string(); // a legal move
        let action = parse_move(&move_string);
        legal_moves.push(action);
        eprintln!("{:?} {:?}", move_string, action);
    }

    let id = if legal_moves[0] == (8, 17) {
        WHITE
    } else {
        BLACK
    };

    eprintln!("ID={}", id);

    let init_start = Instant::now();
    let mut mcts = MCTS::<BitBoardEnv>::with_capacity(id, 2_500_000, default_node_value, 0);
    eprintln!("init time {}ms", init_start.elapsed().as_millis());

    if id == BLACK {
        let opponent_move = parse_move(&opponent_move_string);
        eprintln!("{} {:?}", opponent_move_string, opponent_move);
        mcts.step_action(&opponent_move);
    }

    let (num_steps, millis) = mcts.explore_for(995);
    eprintln!("{} in {}ms", num_steps, millis);

    let action = mcts.best_action();
    let action_str = serialize_move(&action);
    mcts.step_action(&action);
    println!("{}", action_str);
    eprintln!("{} {:?}", action_str, action);

    // game loop
    loop {
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let opponent_move_string = input_line.trim_matches('\n').to_string(); // last move played or "None"
        let opponent_move = parse_move(&opponent_move_string);
        eprintln!("{} {:?}", opponent_move_string, opponent_move);
        let step_start = Instant::now();
        mcts.step_action(&opponent_move);
        let step_elapsed = step_start.elapsed().as_millis();
        eprintln!("{}ms", step_elapsed);

        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let legal_moves = parse_input!(input_line, i32); // number of legal moves
        for _i in 0..legal_moves as usize {
            let mut input_line = String::new();
            io::stdin().read_line(&mut input_line).unwrap();
            let _move_string = input_line.trim().to_string(); // a legal move
        }

        let explore_ms = 95 - step_elapsed;
        let (num_steps, millis) = mcts.explore_for(explore_ms);
        let action = mcts.best_action();
        let action_str = serialize_move(&action);
        println!(
            "{} {} in {}ms | {} / {}",
            action_str,
            num_steps,
            millis,
            mcts.nodes.len(),
            mcts.nodes.capacity()
        );
        mcts.step_action(&action);
    }
}
