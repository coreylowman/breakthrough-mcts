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
