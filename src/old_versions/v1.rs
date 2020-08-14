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

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
enum Piece {
    Empty = 2,
    White = 0,
    Black = 1,
}

type Location = (usize, usize);
type Action = (Location, Location);
type Board = [[Piece; 8]; 8]; // TODO try out single dimensional array

#[derive(Eq, PartialEq, Clone)]
struct PlayerInfo {
    id: usize,
    piece: Piece,
    dy: i32,
    ty: usize,
    pieces_left: i32,
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
                    let from = (x, y);
                    let bny = &self.board[ny];
                    if x > 0 && bny[x - 1] != self.player.piece {
                        acs.push((from, (x - 1, ny)));
                    }
                    if bny[x] == Piece::Empty {
                        acs.push((from, (x, ny)));
                    }
                    if x < 7 && bny[x + 1] != self.player.piece {
                        acs.push((from, (x + 1, ny)));
                    }
                }
            }
        }

        let winning_actions: Vec<Action> = acs
            .iter()
            .cloned()
            .filter(|a| (a.1).1 == self.player.ty)
            .collect();

        if winning_actions.len() > 0 {
            winning_actions
        } else {
            acs
        }
    }

    fn step(&mut self, action: &Action) {
        let &((from_x, from_y), (to_x, to_y)) = action;

        // if self.board[to_y][to_x] == self.opponent.piece {
        //     self.opponent.pieces_left -= 1;
        // }

        self.board[from_y][from_x] = Piece::Empty;
        self.board[to_y][to_x] = self.player.piece;

        if self.player.ty == to_y {
            // || self.opponent.pieces_left == 0 {
            self.player.won = true;
            return;
        }

        std::mem::swap(&mut self.player, &mut self.opponent);
    }
}

struct Node {
    id: usize,
    parent: Option<usize>,
    env: Env,
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
        let env = match parent {
            Some((node, action)) => {
                let mut env = node.env.clone();
                env.step(action);
                env
            }
            None => Env::new(),
        };

        let parent_id = match parent {
            Some((node, _action)) => Option::Some(node.id),
            None => Option::None,
        };

        let my_action = match parent {
            Some((node, _action)) => !node.my_action,
            None => false,
        };

        let is_over = env.is_over();

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
    rng: StdRng,
}

impl MCTS {
    fn new(id: usize) -> MCTS {
        let mut root = Node::new(0, Option::None);
        root.my_action = id == WHITE;
        MCTS {
            id: id,
            root: 0,
            nodes: vec![root],
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

    fn explore_expand(&mut self) -> (usize, u128) {
        let start = Instant::now();
        let mut steps = 0;
        while !self.nodes[self.root].expanded {
            self.explore();
            steps += 1;
        }
        (steps, start.elapsed().as_millis())
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

    fn rollout(&mut self, node_id: usize) -> f32 {
        // assert!(node_id < self.nodes.len());
        let mut env = self.nodes[node_id].env.clone();
        while !env.is_over() {
            // TODO could speed this up if we randomly chose the action to choose beforehand?
            let actions = env.actions();
            let action = actions.choose(&mut self.rng).unwrap();
            env.step(action);
        }
        env.reward(self.id)
    }

    fn backprop(&mut self, leaf_node_id: usize, reward: f32) {
        let mut node_id: usize = leaf_node_id;
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
}

fn parse_move(action_str: &String) -> Action {
    let action: Vec<char> = action_str.chars().collect();
    let from_x = action[0] as i32 - 'a' as i32;
    let from_y = action[1] as i32 - '1' as i32;
    let to_x = action[2] as i32 - 'a' as i32;
    let to_y = action[3] as i32 - '1' as i32;
    (
        (from_x as usize, from_y as usize),
        (to_x as usize, to_y as usize),
    )
}

fn serialize_move(action: &Action) -> String {
    let ((from_x, from_y), (to_x, to_y)) = *action;
    format!(
        "{}{}{}{}",
        (from_x as u8 + 'a' as u8) as char,
        from_y + 1,
        (to_x as u8 + 'a' as u8) as char,
        to_y + 1
    )
}

fn local_main() {
    // let mut env = Env::new();
    let mut white_mcts = MCTS::new(0);
    // let mut black_mcts = MCTS::new(1);

    let (num_steps, millis) = white_mcts.explore_n(50_000);
    eprintln!(
        "{} ({} in {}ms)",
        num_steps as f32 / millis as f32,
        num_steps,
        millis
    );

    // let white_first = white_mcts.best_action();
    // env.step(&white_first);
    // white_mcts.step_action(&white_first);
    // black_mcts.step_action(&white_first);

    // let (num_steps, millis) = black_mcts.explore_for(990);
    // eprintln!(
    //     "{} ({} in {}ms)",
    //     num_steps as f32 / millis as f32,
    //     num_steps,
    //     millis
    // );

    // let black_first = black_mcts.best_action();
    // env.step(&black_first);
    // black_mcts.step_action(&black_first);
    // white_mcts.step_action(&black_first);

    // while !env.is_over() {
    //     let (num_steps, millis) = white_mcts.explore_for(90);
    //     eprintln!(
    //         "{} ({} in {}ms)",
    //         num_steps as f32 / millis as f32,
    //         num_steps,
    //         millis
    //     );
    //     let white_first = white_mcts.best_action();
    //     env.step(&white_first);
    //     white_mcts.step_action(&white_first);
    //     black_mcts.step_action(&white_first);

    //     if env.is_over() {
    //         break;
    //     }

    //     let (num_steps, millis) = black_mcts.explore_for(90);
    //     eprintln!(
    //         "{} ({} in {}ms)",
    //         num_steps as f32 / millis as f32,
    //         num_steps,
    //         millis
    //     );
    //     let black_first = black_mcts.best_action();
    //     env.step(&black_first);
    //     black_mcts.step_action(&black_first);
    //     white_mcts.step_action(&black_first);
    // }

    // println!("{} {}", env.reward(WHITE), env.reward(BLACK));

    // let node = &mcts.nodes[0];
    // println!("{}", node.expanded);
    // println!("{}r {}v", node.reward, node.num_visits);

    // for (k, &v) in node.actions.iter().zip(node.children.iter()) {
    //     println!(
    //         "{:?}: {}r {}v {}",
    //         serialize_move(k),
    //         mcts.nodes[v].reward,
    //         mcts.nodes[v].num_visits,
    //         mcts.nodes[v].reward / (mcts.nodes[v].num_visits as f32)
    //     );
    // }

    // let best = mcts.best_action();
    // println!("{}", serialize_move(&best));
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

    let id = if legal_moves[0] == ((0, 1), (1, 2)) {
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

fn main() {
    local_main();
    // codingame_main();
}
