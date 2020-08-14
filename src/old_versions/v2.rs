extern crate rand;

use crate::rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use std::io;
use std::time::Instant;

macro_rules! parse_input {
    ($x:expr, $t:ident) => {
        $x.trim().parse::<$t>().unwrap()
    };
}

const WHITE: usize = 0;
const BLACK: usize = 1;

type BitBoard = u64;
type Square = u32;
type Action = (Square, Square);

const COL_1: BitBoard = 0x0101010101010101u64;
const NOT_COL_1: BitBoard = !COL_1;
const COL_8: BitBoard = COL_1 << 7;
const NOT_COL_8: BitBoard = !COL_8;
const ROW_1: BitBoard = 0xFFu64;
const ROW_2: BitBoard = ROW_1 << 8;
const ROW_7: BitBoard = ROW_1 << 48;
const ROW_8: BitBoard = ROW_1 << 56;
const FULL_BOARD: BitBoard = 0xFFFFFFFFFFFFFFFFu64;

struct BitBoardIterator(BitBoard);

impl Iterator for BitBoardIterator {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }

        let sq = self.0.trailing_zeros();
        self.0 &= self.0.wrapping_sub(1);

        Some(sq)
    }
}

struct ActionIterator(BitBoard, BitBoard);

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
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
enum Piece {
    Empty = 2,
    White = 0,
    Black = 1,
}

type Board = [[Piece; 8]; 8]; // TODO try out single dimensional array

#[derive(Eq, PartialEq, Clone)]
struct PlayerInfo {
    id: usize,
    piece: Piece,
    dy: i32,
    ty: u32,
    pieces_left: u64,
    won: bool,
}

#[derive(Eq, PartialEq, Clone)]
struct Env {
    player: PlayerInfo,
    opponent: PlayerInfo,
    board: Board,
}

impl Env {
    fn new() -> Env {
        let mut board = [[Piece::Empty; 8]; 8];
        for x in 0..8 {
            board[0][x] = Piece::White;
            board[1][x] = Piece::White;
            board[6][x] = Piece::Black;
            board[7][x] = Piece::Black;
        }

        Env {
            player: PlayerInfo {
                id: WHITE,
                piece: Piece::White,
                dy: 1,
                ty: 7,
                pieces_left: 16,
                won: false,
            },
            opponent: PlayerInfo {
                id: BLACK,
                piece: Piece::Black,
                dy: -1,
                ty: 0,
                pieces_left: 16,
                won: false,
            },
            board: board,
        }
    }

    fn is_over(&self) -> bool {
        self.player.won
    }

    fn reward(&self, color: usize) -> f32 {
        // assert!(self.player.won);
        if self.player.piece as usize == color {
            1.0
        } else {
            0.0 // TODO test out -1.0
        }
    }

    fn actions(&self) -> Vec<Action> {
        let mut acs = Vec::with_capacity(16 * 3);

        for y in 0..8 {
            let ny = (y as i32 + self.player.dy) as usize;
            for x in 0..8 {
                if self.board[y][x] == self.player.piece {
                    let from = (y * 8 + x) as u32;
                    let bny = &self.board[ny];
                    if x > 0 && bny[x - 1] != self.player.piece {
                        acs.push((from, (ny * 8 + x - 1) as u32));
                    }
                    if bny[x] == Piece::Empty {
                        acs.push((from, (ny * 8 + x) as u32));
                    }
                    if x < 7 && bny[x + 1] != self.player.piece {
                        acs.push((from, (ny * 8 + x + 1) as u32));
                    }
                }
            }
        }

        let winning_actions: Vec<Action> = acs
            .iter()
            .cloned()
            .filter(|a| a.1 / 8 == self.player.ty)
            .collect();

        if winning_actions.len() > 0 {
            winning_actions
        } else {
            acs
        }
    }

    fn step(&mut self, action: &Action) -> bool {
        let &(from, to) = action;
        let from_y = (from / 8) as usize;
        let from_x = (from % 8) as usize;
        let to_y = (to / 8) as usize;
        let to_x = (to % 8) as usize;

        if self.board[to_y][to_x] == self.opponent.piece {
            self.opponent.pieces_left -= 1;
        }

        self.board[from_y][from_x] = Piece::Empty;
        self.board[to_y][to_x] = self.player.piece;

        if self.player.ty == to_y as u32 || self.opponent.pieces_left == 0 {
            self.player.won = true;
            true
        } else {
            std::mem::swap(&mut self.player, &mut self.opponent);
            false
        }
    }
}

#[derive(Eq, PartialEq, Clone)]
struct BitBoardPlayerInfo {
    id: usize,
    pieces: BitBoard,
    fwd_shift: u32,
    right_shift: u32,
    left_shift: u32,
    pieces_left: u64,
    won: bool,
    ty: BitBoard,
}

#[derive(Eq, PartialEq, Clone)]
struct BitBoardEnv {
    player: BitBoardPlayerInfo,
    opponent: BitBoardPlayerInfo,
}

impl BitBoardEnv {
    fn new() -> BitBoardEnv {
        BitBoardEnv {
            player: BitBoardPlayerInfo {
                id: WHITE,
                pieces: ROW_1 | ROW_2,
                left_shift: 7,
                fwd_shift: 8,
                right_shift: 9,
                pieces_left: 16,
                won: false,
                ty: ROW_8,
            },
            opponent: BitBoardPlayerInfo {
                id: BLACK,
                pieces: ROW_7 | ROW_8,
                left_shift: 55,
                fwd_shift: 56,
                right_shift: 57,
                pieces_left: 16,
                won: false,
                ty: ROW_1,
            },
        }
    }

    // #[inline(always)]
    fn is_over(&self) -> bool {
        self.opponent.won
    }

    // #[inline(always)]
    fn reward(&self, color: usize) -> f32 {
        // assert!(self.opponent.won);
        if self.opponent.id == color {
            1.0
        } else {
            0.0 // TODO test out -1.0
        }
    }

    fn actions(&self) -> Vec<(Square, Square)> {
        let mut acs = Vec::with_capacity(16 * 3);

        let p = &self.player;

        let pieces = p.pieces;
        let npieces = !pieces;
        let empty_squares = npieces & !self.opponent.pieces;

        let fwd_to = pieces.rotate_left(p.fwd_shift) & empty_squares;
        let right_to = (pieces & NOT_COL_8).rotate_left(p.right_shift) & npieces;
        let left_to = (pieces & NOT_COL_1).rotate_left(p.left_shift) & npieces;

        let fwd_win = fwd_to & p.ty;
        let right_win = right_to & p.ty;
        let left_win = left_to & p.ty;

        let raw_any_win = fwd_win != 0 || right_win != 0 || left_win != 0;
        let any_win = raw_any_win as u64;
        let none_win = (!raw_any_win) as u64;

        let fwd_to = (any_win * fwd_win) | (none_win * fwd_to);
        let right_to = (any_win * right_win) | (none_win * right_to);
        let left_to = (any_win * left_win) | (none_win * left_to);

        let fwd_from = fwd_to.rotate_left(64 - p.fwd_shift);
        let right_from = right_to.rotate_left(64 - p.right_shift);
        let left_from = left_to.rotate_left(64 - p.left_shift);

        acs.extend(ActionIterator(fwd_from, fwd_to));
        acs.extend(ActionIterator(right_from, right_to));
        acs.extend(ActionIterator(left_from, left_to));
        acs
    }

    fn get_random_action(&self, rng: &mut StdRng) -> Action {
        let p = &self.player;

        let pieces = p.pieces;
        let npieces = !pieces;
        let empty_squares = npieces & !self.opponent.pieces;

        let fwd_to = pieces.rotate_left(p.fwd_shift) & empty_squares;
        let right_to = (pieces & NOT_COL_8).rotate_left(p.right_shift) & npieces;
        let left_to = (pieces & NOT_COL_1).rotate_left(p.left_shift) & npieces;

        let fwd_win = fwd_to & p.ty;
        let right_win = right_to & p.ty;
        let left_win = left_to & p.ty;

        let raw_any_win = fwd_win != 0 || right_win != 0 || left_win != 0;
        let any_win = raw_any_win as u64;
        let none_win = (!raw_any_win) as u64;

        let fwd_to = (any_win * fwd_win) | (none_win * fwd_to);
        let right_to = (any_win * right_win) | (none_win * right_to);
        let left_to = (any_win * left_win) | (none_win * left_to);

        let num_fwd_acs = fwd_to.count_ones();
        let num_right_acs = right_to.count_ones();
        let num_left_acs = left_to.count_ones();

        let i = rng.gen_range(0, num_fwd_acs + num_right_acs + num_left_acs);

        if i >= num_fwd_acs + num_right_acs {
            // generate a left action
            let left_from = left_to.rotate_left(64 - p.left_shift);
            ActionIterator(left_from, left_to)
                .nth((i - (num_fwd_acs + num_right_acs)) as usize)
                .unwrap()
        } else if i >= num_fwd_acs {
            // generate a right action
            let right_from = right_to.rotate_left(64 - p.right_shift);
            ActionIterator(right_from, right_to)
                .nth((i - num_fwd_acs) as usize)
                .unwrap()
        } else {
            // generate a forward action
            let fwd_from = fwd_to.rotate_left(64 - p.fwd_shift);
            ActionIterator(fwd_from, fwd_to).nth(i as usize).unwrap()
        }

        // let fwd_from = fwd_to.rotate_left(64 - p.fwd_shift);
        // let right_from = right_to.rotate_left(64 - p.right_shift);
        // let left_from = left_to.rotate_left(64 - p.left_shift);

        // ActionIterator(fwd_from, fwd_to)
        //     .chain(ActionIterator(right_from, right_to))
        //     .chain(ActionIterator(left_from, left_to))
        //     .nth(i as usize)
        //     .unwrap()
    }

    // #[inline(never)]
    fn step(&mut self, action: &(Square, Square)) -> bool {
        let &(from_sq, to_sq) = action;

        let to = 1 << to_sq;
        let from = 1 << from_sq;

        self.opponent.pieces_left -= (self.opponent.pieces >> to_sq) & 1;
        self.opponent.pieces &= !to;
        self.player.pieces = (self.player.pieces | to) & !from;

        // note: count_ones() here is slower than keeping track ourselves
        self.player.won = self.player.ty & to != 0 || self.opponent.pieces_left == 0;

        std::mem::swap(&mut self.player, &mut self.opponent);

        self.opponent.won
    }
}

struct Node {
    id: usize,
    parent: Option<usize>,
    env: BitBoardEnv,
    terminal: bool,
    expanded: bool,
    my_action: bool,
    actions: Vec<Action>,
    children: Vec<usize>,
    unvisited_actions: Vec<Action>,
    num_visits: f32,
    reward: f32,
}

impl Node {
    fn new(id: usize, parent: Option<(&Node, &Action)>) -> Node {
        let (env, parent_id, my_action, is_over) = match parent {
            Some((node, action)) => {
                let mut env = node.env.clone();
                env.step(action);
                let is_over = env.is_over();
                (env, Some(node.id), !node.my_action, is_over)
            }
            None => (BitBoardEnv::new(), None, false, false),
        };

        let actions = if is_over { Vec::new() } else { env.actions() };

        Node {
            id: id,
            parent: parent_id,
            env: env,
            terminal: is_over,
            expanded: false,
            my_action: my_action,
            actions: Vec::new(),
            children: Vec::new(),
            unvisited_actions: actions,
            num_visits: 0.0,
            reward: 0.0,
        }
    }
}

struct MCTS {
    id: usize,
    root: usize,
    nodes: Vec<Node>,
    rng: StdRng, // note: this is about the same performance as SmallRng or any of the XorShiftRngs that got moved to the xorshift crate
}

impl MCTS {
    fn new(id: usize) -> MCTS {
        let mut root = Node::new(0, Option::None);
        root.my_action = id == WHITE;
        // let mut nodes = Vec::with_capacity(1_000_000); // TODO tune this??
        let mut nodes = Vec::new();
        nodes.push(root);
        MCTS {
            id: id,
            root: 0,
            nodes: nodes,
            rng: StdRng::seed_from_u64(0),
        }
    }

    fn best_action(&self) -> Action {
        let root = &self.nodes[self.root];

        let mut best_action_ind = 0;
        let mut best_value = -std::f32::INFINITY;

        for (i, &child_id) in root.children.iter().enumerate() {
            let child = &self.nodes[child_id];
            let value = child.reward / child.num_visits; // TODO test out just child.reward
            if value > best_value {
                best_value = value;
                best_action_ind = i;
            }
        }

        root.actions[best_action_ind].clone()
    }

    fn step_action(&mut self, action: &Action) {
        self.root = match self.nodes[self.root]
            .actions
            .iter()
            .position(|a| a == action)
        {
            Option::Some(action_index) => self.nodes[self.root].children[action_index],
            Option::None => {
                let node_id = self.nodes.len();
                self.nodes.push(Node::new(
                    node_id,
                    Option::Some((&self.nodes[self.root], action)),
                ));
                node_id
            }
        };
    }

    fn explore_for(&mut self, millis: u128) -> (usize, u128) {
        let start = Instant::now();
        let mut steps = 0;
        while start.elapsed().as_millis() < millis {
            self.explore();
            steps += 1;
        }
        (steps, start.elapsed().as_millis())
    }

    fn explore_n(&mut self, n: usize) -> (usize, u128) {
        let start = Instant::now();
        for _i in 0..n {
            self.explore();
        }
        (n, start.elapsed().as_millis())
    }

    fn time_explore_n(&mut self, n: usize) -> (usize, u128) {
        let mut select_ns = 0;
        let mut rollout_ns = 0;
        let mut get_action_ns = 0;
        let mut step_ns = 0;
        let mut rn = 0;
        let mut backprop_ns = 0;

        let start = Instant::now();
        for _i in 0..n {
            let select_start = Instant::now();
            let node_id = self.select_node();
            select_ns += select_start.elapsed().as_nanos();

            let rollout_start = Instant::now();
            let reward = {
                let mut env = self.nodes[node_id].env.clone();
                let mut is_over = env.is_over();
                while !is_over {
                    rn += 1;

                    let get_start = Instant::now();
                    let action = env.get_random_action(&mut self.rng);
                    get_action_ns += get_start.elapsed().as_nanos();

                    let step_start = Instant::now();
                    is_over = env.step(&action);
                    step_ns += step_start.elapsed().as_nanos();
                }
                env.reward(self.id)
            };
            rollout_ns += rollout_start.elapsed().as_nanos();

            let backprop_start = Instant::now();
            self.backprop(node_id, reward);
            backprop_ns += backprop_start.elapsed().as_nanos();
        }

        println!(
            "{} {} {}",
            select_ns as f32 / n as f32,
            rollout_ns as f32 / n as f32,
            backprop_ns as f32 / n as f32
        );
        println!(
            "{} {}",
            get_action_ns as f32 / rn as f32,
            step_ns as f32 / rn as f32,
        );
        (n, start.elapsed().as_millis())
    }

    // #[inline(always)]
    fn explore(&mut self) {
        let node_id = self.select_node();
        let reward = self.rollout(node_id);
        self.backprop(node_id, reward);
    }

    fn select_node(&mut self) -> usize {
        let mut node_id = self.root;
        loop {
            // assert!(node_id < self.nodes.len());
            let node = &self.nodes[node_id];
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

        let node = &self.nodes[node_id];

        let mut best_child = 0;
        let mut best_value = -std::f32::INFINITY;

        let parent_visits = node.num_visits.log(2.0);

        for &child_id in node.children.iter() {
            let child = &self.nodes[child_id];

            // TODO test out different value functions here
            let value =
                child.reward / child.num_visits + (2.0 * parent_visits / child.num_visits).sqrt();

            if value > best_value {
                best_value = value;
                best_child = child_id;
            }
        }

        best_child
    }

    fn select_unexpanded_child(&mut self, node_id: usize) -> usize {
        let child_id = self.nodes.len();

        let child_node = {
            let node = &mut self.nodes[node_id];
            let action = node.unvisited_actions.pop().unwrap();
            if node.unvisited_actions.len() == 0 {
                node.expanded = true;
            }
            node.actions.push(action);
            node.children.push(child_id);
            Node::new(child_id, Option::Some((&node, &action)))
        };

        self.nodes.push(child_node);

        child_id
    }

    fn rollout(&mut self, node_id: usize) -> f32 {
        // assert!(node_id < self.nodes.len());
        let mut env = self.nodes[node_id].env.clone();
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

            let node = &mut self.nodes[node_id];

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

            node_id = match node.parent {
                Some(node_id) => node_id,
                None => break,
            };
        }
    }
}

fn parse_move(action_str: &String) -> Action {
    let action: Vec<char> = action_str.chars().collect();
    let from_x = action[0] as i32 - 'a' as i32;
    let from_y = action[1] as i32 - '1' as i32;
    let to_x = action[2] as i32 - 'a' as i32;
    let to_y = action[3] as i32 - '1' as i32;
    ((from_y * 8 + from_x) as u32, (to_y * 8 + to_x) as u32)
}

fn serialize_move(action: &Action) -> String {
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

fn codingame_main() {
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

    let mut mcts = MCTS::new(id);

    if id == BLACK {
        let opponent_move = parse_move(&opponent_move_string);
        eprintln!("{} {:?}", opponent_move_string, opponent_move);
        mcts.step_action(&opponent_move);
    }

    let (num_steps, millis) = mcts.explore_for(995);
    eprintln!("{} in {}us", num_steps, millis);

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
        mcts.step_action(&opponent_move);

        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let legal_moves = parse_input!(input_line, i32); // number of legal moves
        for _i in 0..legal_moves as usize {
            let mut input_line = String::new();
            io::stdin().read_line(&mut input_line).unwrap();
            let _move_string = input_line.trim().to_string(); // a legal move
        }

        let (num_steps, millis) = mcts.explore_for(98);
        let action = mcts.best_action();
        let action_str = serialize_move(&action);
        println!("{}", action_str);
        mcts.step_action(&action);
        eprintln!("{} in {}us", num_steps, millis);
        eprintln!("{} {:?}", action_str, action);
    }
}

#[test]
fn test_iter_rows() {
    for y in 0..8u32 {
        let locs: Vec<Square> = BitBoardIterator(ROW_1 << y * 8).collect();
        assert_eq!(locs.len(), 8);
        for i in 0..8u32 {
            assert_eq!(locs[i as usize], y * 8 + i);
        }
    }
}

#[test]
fn test_iter_cols() {
    for x in 0..8 {
        let locs: Vec<Square> = BitBoardIterator(COL_1 << x).collect();
        assert_eq!(locs.len(), 8);
        for y in 0..8 {
            assert_eq!(locs[y], (y * 8 + x) as u32);
        }
    }
}

#[test]
fn test_iter_empty() {
    let locs: Vec<Square> = BitBoardIterator(0).collect();
    assert_eq!(locs.len(), 0);
}

#[test]
fn test_iter_full() {
    let locs: Vec<Square> = BitBoardIterator(FULL_BOARD).collect();
    assert_eq!(locs.len(), 64);
    for y in 0..8 {
        for x in 0..8 {
            let i = y * 8 + x;
            assert_eq!(locs[i], i as u32);
        }
    }
}

#[test]
fn test_start_white_actions() {
    let true_env = Env::new();

    let env = BitBoardEnv::new();
    // assert_eq!(env.count_actions(), 22);
    let mut acs = env.actions();
    let mut true_acs = true_env.actions();

    assert_eq!(acs.len(), 22);
    assert_eq!(acs.len(), true_acs.len());

    acs.sort();
    true_acs.sort();

    for &true_a in true_acs.iter() {
        assert!(acs.iter().position(|&a| a == true_a).is_some());
    }
}

#[test]
fn test_start_black_actions() {
    let mut true_env = Env::new();
    std::mem::swap(&mut true_env.player, &mut true_env.opponent);

    let mut env = BitBoardEnv::new();
    std::mem::swap(&mut env.player, &mut env.opponent);

    // assert_eq!(env.count_actions(), 22);
    let mut acs = env.actions();
    let mut true_acs = true_env.actions();

    assert_eq!(acs.len(), 22);
    assert_eq!(acs.len(), true_acs.len());

    acs.sort();
    true_acs.sort();

    for ac in acs.iter() {
        println!("{:?}", ac);
    }
    println!();

    for ac in true_acs.iter() {
        println!("{:?}", ac);
    }
    println!();

    for &true_a in true_acs.iter() {
        assert!(acs.iter().position(|&a| a == true_a).is_some());
    }
}

#[test]
fn test_to_squares() {
    let env = BitBoardEnv::new();
    let p = &env.player;

    let empty_squares = !p.pieces & !env.opponent.pieces;

    let fwd_to = p.pieces.rotate_left(p.fwd_shift) & empty_squares;
    assert_eq!(BitBoardIterator(fwd_to).collect::<Vec<Square>>().len(), 8);
    for (i, sq) in BitBoardIterator(fwd_to).enumerate() {
        let y = sq / 8;
        let x = sq % 8;
        assert_eq!((x, y), (i as u32, 2));
    }

    let right_to = (p.pieces & NOT_COL_8).rotate_left(p.right_shift) & !p.pieces;
    assert_eq!(BitBoardIterator(right_to).collect::<Vec<Square>>().len(), 7);
    for (i, sq) in BitBoardIterator(right_to).enumerate() {
        let y = sq / 8;
        let x = sq % 8;
        assert_eq!((x, y), (i as u32 + 1, 2));
    }

    let left_to = (p.pieces & NOT_COL_1).rotate_left(p.left_shift) & !p.pieces;
    assert_eq!(BitBoardIterator(left_to).collect::<Vec<Square>>().len(), 7);
    for (i, sq) in BitBoardIterator(left_to).enumerate() {
        let y = sq / 8;
        let x = sq % 8;
        assert_eq!((x, y), (i as u32, 2));
    }
}

#[test]
fn test_from_squares() {
    let env = BitBoardEnv::new();
    let p = &env.player;

    let empty_squares = !p.pieces & !env.opponent.pieces;

    let fwd_to = p.pieces.rotate_left(p.fwd_shift) & empty_squares;
    let right_to = (p.pieces & NOT_COL_8).rotate_left(p.right_shift) & !p.pieces;
    let left_to = (p.pieces & NOT_COL_1).rotate_left(p.left_shift) & !p.pieces;

    let fwd_from = fwd_to.rotate_left(64 - p.fwd_shift);
    let right_from = right_to.rotate_left(64 - p.right_shift);
    let left_from = left_to.rotate_left(64 - p.left_shift);

    assert_eq!(BitBoardIterator(fwd_from).collect::<Vec<Square>>().len(), 8);
    for (i, sq) in BitBoardIterator(fwd_from).enumerate() {
        let y = sq / 8;
        let x = sq % 8;
        assert_eq!((x, y), (i as u32, 1));
    }

    assert_eq!(
        BitBoardIterator(right_from).collect::<Vec<Square>>().len(),
        7
    );
    for (i, sq) in BitBoardIterator(right_from).enumerate() {
        let y = sq / 8;
        let x = sq % 8;
        assert_eq!((x, y), (i as u32, 1));
    }

    assert_eq!(
        BitBoardIterator(left_from).collect::<Vec<Square>>().len(),
        7
    );
    for (i, sq) in BitBoardIterator(left_from).enumerate() {
        let y = sq / 8;
        let x = sq % 8;
        assert_eq!((x, y), (i as u32 + 1, 1));
    }
}

#[test]
fn step_white() {
    let mut env = BitBoardEnv::new();

    env.step(&(8, 16));

    assert_eq!((env.opponent.pieces >> 16) & 1, 1);
    assert_eq!((env.opponent.pieces >> 8) & 1, 0);

    env.step(&(48, 40));
    assert_eq!((env.opponent.pieces >> 48) & 1, 0);
    assert_eq!((env.opponent.pieces >> 40) & 1, 1);
}

#[test]
fn test_random_game() {
    let mut raw_env = Env::new();
    let mut bb_env = BitBoardEnv::new();

    let mut rng = StdRng::seed_from_u64(0);

    let mut i = 0;
    while !raw_env.is_over() {
        i += 1;
        println!("{}", i);
        assert_eq!(bb_env.is_over(), raw_env.is_over());

        // let bb_num_acs = bb_env.count_actions();
        let mut bb_acs = bb_env.actions();
        let mut raw_acs = raw_env.actions();

        bb_acs.sort();
        raw_acs.sort();

        // assert_eq!(bb_num_acs as usize, bb_acs.len());
        assert_eq!(bb_acs.len(), raw_acs.len());
        for &true_a in raw_acs.iter() {
            assert!(bb_acs.iter().position(|&a| a == true_a).is_some());
        }

        let ac = raw_acs.choose(&mut rng).unwrap();

        let bb_is_over = bb_env.step(&ac);
        let raw_is_over = raw_env.step(&ac);

        assert_eq!(bb_is_over, raw_is_over);
        assert_eq!(bb_env.is_over(), raw_env.is_over());

        // assert_eq!(bb_env.player.id, raw_env.player.id);
        // assert_eq!(bb_env.opponent.id, raw_env.opponent.id);
        // assert_eq!(bb_env.player.pieces_left, raw_env.player.pieces_left);
        // assert_eq!(bb_env.opponent.pieces_left, raw_env.opponent.pieces_left);
    }

    assert_eq!(bb_env.is_over(), raw_env.is_over());
    assert_eq!(bb_env.reward(0), raw_env.reward(0));
    assert_eq!(bb_env.reward(1), raw_env.reward(1));
}

fn local_main() {
    // let mut env = Env::new();
    let mut white_mcts = MCTS::new(0);
    // let mut black_mcts = MCTS::new(1);

    // let (num_steps, millis) = white_mcts.time_explore_n(50_000);
    let (num_steps, millis) = white_mcts.explore_for(1000);
    eprintln!(
        "{} ({} in {}ms)... {} nodes",
        num_steps as f32 / millis as f32,
        num_steps,
        millis,
        white_mcts.nodes.len(),
    );
}

fn main() {
    local_main();
    // codingame_main();
}
