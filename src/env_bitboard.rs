use crate::env::{Env, BLACK, WHITE};

use crate::rand::rngs::StdRng;
use crate::rand::Rng;

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

pub struct BitBoardIterator(BitBoard);

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

pub struct ActionIterator(BitBoard, BitBoard);

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

#[derive(Eq, PartialEq, Clone)]
pub struct PlayerInfo {
    pub id: usize,
    pieces: BitBoard,
    fwd_shift: u32,
    right_shift: u32,
    left_shift: u32,
    pieces_left: u64,
    won: bool,
    ty: BitBoard,
}

#[derive(Eq, PartialEq, Clone)]
pub struct BitBoardEnv {
    pub player: PlayerInfo,
    pub opponent: PlayerInfo,
}

impl BitBoardEnv {
    fn action_bitboards(&self) -> (BitBoard, BitBoard, BitBoard) {
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

        (fwd_to, right_to, left_to)
    }
}

impl Env for BitBoardEnv {
    type Action = (Square, Square);

    fn new() -> BitBoardEnv {
        BitBoardEnv {
            player: PlayerInfo {
                id: WHITE,
                pieces: ROW_1 | ROW_2,
                left_shift: 7,
                fwd_shift: 8,
                right_shift: 9,
                pieces_left: 16,
                won: false,
                ty: ROW_8,
            },
            opponent: PlayerInfo {
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

    fn turn(&self) -> usize {
        self.player.id
    }

    fn is_over(&self) -> bool {
        self.opponent.won
    }

    fn reward(&self, color: usize) -> f32 {
        // assert!(self.opponent.won);
        if self.opponent.id == color {
            1.0
        } else {
            0.0 // TODO test out -1.0
        }
    }

    fn actions(&self) -> Vec<Self::Action> {
        let mut acs = Vec::with_capacity(16 * 3);

        let p = &self.player;

        let (fwd_to, right_to, left_to) = self.action_bitboards();

        let fwd_from = fwd_to.rotate_left(64 - p.fwd_shift);
        let right_from = right_to.rotate_left(64 - p.right_shift);
        let left_from = left_to.rotate_left(64 - p.left_shift);

        acs.extend(ActionIterator(fwd_from, fwd_to));
        acs.extend(ActionIterator(right_from, right_to));
        acs.extend(ActionIterator(left_from, left_to));
        acs
    }

    fn get_random_action(&self, rng: &mut StdRng) -> Self::Action {
        let p = &self.player;

        let (fwd_to, right_to, left_to) = self.action_bitboards();

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
    }

    fn step(&mut self, action: &Self::Action) -> bool {
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
