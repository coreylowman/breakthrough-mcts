extern crate rand;

mod env;
mod env_bitboard;
// mod env_naive;

mod mcts;

use std::io;
use std::time::Instant;

use env::{Env, BLACK, WHITE};
use env_bitboard::{BitBoardEnv, PlayerInfo};
use mcts::{Node, MCTS};

macro_rules! parse_input {
    ($x:expr, $t:ident) => {
        $x.trim().parse::<$t>().unwrap()
    };
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

    let init_start = Instant::now();
    let mut mcts = MCTS::<BitBoardEnv>::with_capacity(id, 2_500_000, 0);
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

fn first_explore() {
    let mut white_mcts = MCTS::<BitBoardEnv>::with_capacity(WHITE, 2_000_000, 0);

    let (num_steps, millis) = white_mcts.explore_n(100_000);
    eprintln!(
        "{} ({} in {}ms)... {} nodes",
        num_steps as f32 / millis as f32,
        num_steps,
        millis,
        white_mcts.nodes.len(),
    );
}

fn timed_first_explore() {
    let mut white_mcts = MCTS::<BitBoardEnv>::with_capacity(WHITE, 2_000_000, 0);

    let (num_steps, millis) = white_mcts.timed_explore_n(1_000_000);
    eprintln!(
        "{} ({} in {}ms)... {} nodes",
        num_steps as f32 / millis as f32,
        num_steps,
        millis,
        white_mcts.nodes.len(),
    );
}

fn run_game() {
    let mut env = BitBoardEnv::new();

    let mut white_total_ms = 0;
    let mut white_step_ms = 0;
    let mut black_total_ms = 0;
    let mut black_step_ms = 0;

    let white_init_start = Instant::now();
    let mut white_mcts = MCTS::<BitBoardEnv>::with_capacity(WHITE, 1_500_000, 0);
    println!("white {}ms", white_init_start.elapsed().as_millis());

    let black_init_start = Instant::now();
    let mut black_mcts = MCTS::<BitBoardEnv>::with_capacity(BLACK, 1_500_000, 0);
    println!("black {}ms", black_init_start.elapsed().as_millis());

    let (num_steps, millis) = white_mcts.explore_n(300_000);
    white_total_ms += millis;
    eprintln!("{} in {}ms", num_steps, millis);

    let mut action = white_mcts.best_action();
    println!("{:?}", action);
    env.step(&action);

    let mut step_start = Instant::now();
    white_mcts.step_action(&action);
    let mut step_elapsed = step_start.elapsed().as_millis();
    // println!("{}ms", step_elapsed);
    white_step_ms += step_elapsed;

    step_start = Instant::now();
    black_mcts.step_action(&action);
    step_elapsed = step_start.elapsed().as_millis();
    // println!("{}ms", step_elapsed);
    black_step_ms += step_elapsed;

    let (num_steps, millis) = black_mcts.explore_n(300_000);
    black_total_ms += millis;
    eprintln!("{} in {}ms", num_steps, millis);

    action = black_mcts.best_action();
    println!("{:?}", action);

    env.step(&action);
    step_start = Instant::now();
    white_mcts.step_action(&action);
    step_elapsed = step_start.elapsed().as_millis();
    println!("{}ms", step_elapsed);
    white_step_ms += step_elapsed;

    step_start = Instant::now();
    black_mcts.step_action(&action);
    step_elapsed = step_start.elapsed().as_millis();
    println!("{}ms", step_elapsed);
    black_step_ms += step_elapsed;

    let mut i = 0;
    while !env.is_over() {
        i += 1;
        println!("{}", i);

        let action = if env.turn() == WHITE {
            let pre_node_size = white_mcts.nodes.len();
            let pre_node_capacity = white_mcts.nodes.capacity();
            let (num_steps, millis) = white_mcts.explore_n(50_000);
            white_total_ms += millis;
            let post_node_size = white_mcts.nodes.len();
            let post_node_capacity = white_mcts.nodes.capacity();
            eprintln!(
                "WHITE {}% | {} in {}us | {} / {} -> {} / {}... {}% unused",
                100.0 * (1.0 - white_mcts.nodes[0].reward / white_mcts.nodes[0].num_visits),
                num_steps,
                millis,
                pre_node_size,
                pre_node_capacity,
                post_node_size,
                post_node_capacity,
                100.0 * (1.0 - (post_node_size as f32) / post_node_capacity as f32),
            );

            // if millis > 100 {
            //     panic!("too long")
            // }

            white_mcts.best_action()
        } else {
            let pre_node_size = black_mcts.nodes.len();
            let pre_node_capacity = black_mcts.nodes.capacity();
            let (num_steps, millis) = black_mcts.explore_n(50_000);
            black_total_ms += millis;
            let post_node_size = black_mcts.nodes.len();
            let post_node_capacity = black_mcts.nodes.capacity();
            eprintln!(
                "BLACK {}% | {} in {}us | {} / {} -> {} / {}... {}% unused",
                100.0 * (1.0 - black_mcts.nodes[0].reward / black_mcts.nodes[0].num_visits),
                num_steps,
                millis,
                pre_node_size,
                pre_node_capacity,
                post_node_size,
                post_node_capacity,
                100.0 * (1.0 - (post_node_size as f32) / post_node_capacity as f32),
            );

            // if millis > 100 {
            //     panic!("too long")
            // }

            black_mcts.best_action()
        };

        println!("{:?}", action);

        env.step(&action);
        step_start = Instant::now();
        white_mcts.step_action(&action);
        step_elapsed = step_start.elapsed().as_millis();
        // println!("{}ms", step_elapsed);
        white_step_ms += step_elapsed;

        step_start = Instant::now();
        black_mcts.step_action(&action);
        step_elapsed = step_start.elapsed().as_millis();
        // println!("{}ms", step_elapsed);
        black_step_ms += step_elapsed;
    }

    println!("{} {}", env.reward(WHITE), env.reward(BLACK));

    let white_mb = white_mcts.memory_usage() / 1_000_000;
    let black_mb = black_mcts.memory_usage() / 1_000_000;

    println!(
        "{}mb ({} mb/turn) {}mb ({} mb/turn) | {}ms {}ms | {}ms {}ms",
        white_mb,
        white_mb as f32 / i as f32,
        black_mb,
        black_mb as f32 / i as f32,
        white_total_ms,
        black_total_ms,
        white_step_ms,
        black_step_ms,
    );
}

fn compare<E: Env + Clone>(
    seed: u64,
    white_action_fn: fn(&MCTS<E>) -> E::Action,
    black_action_fn: fn(&MCTS<E>) -> E::Action,
) -> bool {
    let mut env = E::new();
    let mut white_mcts = MCTS::<E>::with_capacity(WHITE, 1_500_000, seed);
    let mut black_mcts = MCTS::<E>::with_capacity(BLACK, 1_500_000, seed);

    let (num_steps, millis) = white_mcts.explore_n(100_000);
    let mut action = white_action_fn(&white_mcts);
    env.step(&action);
    white_mcts.step_action(&action);
    black_mcts.step_action(&action);

    let (num_steps, millis) = black_mcts.explore_n(100_000);
    action = black_action_fn(&black_mcts);
    env.step(&action);
    white_mcts.step_action(&action);
    black_mcts.step_action(&action);

    while !env.is_over() {
        let action = if env.turn() == WHITE {
            let (num_steps, millis) = white_mcts.explore_n(30_000);
            white_action_fn(&white_mcts)
        } else {
            let (num_steps, millis) = black_mcts.explore_n(30_000);
            black_action_fn(&black_mcts)
        };

        env.step(&action);
        white_mcts.step_action(&action);
        black_mcts.step_action(&action);
    }

    if env.reward(WHITE) > env.reward(BLACK) {
        WHITE
    } else {
        BLACK
    }
}

fn local_main() {
    println!("Node {}", std::mem::size_of::<Node<BitBoardEnv>>());
    println!("BitBoardEnv {}", std::mem::size_of::<BitBoardEnv>());
    println!("PlayerInfo {}", std::mem::size_of::<PlayerInfo>());
    println!(
        "BitBoardEnv::ActionIterator {}",
        std::mem::size_of::<<env_bitboard::BitBoardEnv as Env>::ActionIterator>()
    );
    first_explore();
    println!();
    // timed_first_explore();
    // run_game();

    // 300 games | WHITE 182 (60.67%) | BLACK 118 (39.33%)
    // 300 games | WHITE 137 (45.67%) | BLACK 163 (54.33%)
    // let mut wins = [0, 0];
    // for i in 0..300 {
    //     let winner = compare::<BitBoardEnv>(
    //         i,
    //         |mcts: &MCTS<BitBoardEnv>| mcts.best_action(),
    //         |mcts: &MCTS<BitBoardEnv>| mcts.negamax(2),
    //     );
    //     wins[winner as usize] += 1;
    //     println!(
    //         "{:03} games | WHITE {:03} ({:05.2}%) | BLACK {:03} ({:05.2}%)",
    //         i + 1,
    //         wins[WHITE as usize],
    //         100.0 * wins[WHITE as usize] as f32 / (i as f32 + 1.0),
    //         wins[BLACK as usize],
    //         100.0 * wins[BLACK as usize] as f32 / (i as f32 + 1.0)
    //     );
    // }
}

fn main() {
    local_main();
    // codingame_main();
}
