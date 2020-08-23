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

impl ActionIterator {
    // TODO how is this slower???
    #[inline]
    fn fast_nth(&mut self, n: u32) -> (Square, Square) {
        for _ in 0..n {
            self.0 &= self.0.wrapping_sub(1);
            self.1 &= self.1.wrapping_sub(1);
        }
        (self.0.trailing_zeros(), self.1.trailing_zeros())
    }
}

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
    pieces_left: u8,
    ty: BitBoard,
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

        let fwd_win = fwd_to & self.me.ty;
        let right_win = right_to & self.me.ty;
        let left_win = left_to & self.me.ty;

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

    fn new() -> BitBoardEnv {
        BitBoardEnv {
            my_bb: ROW_1 | ROW_2,
            op_bb: ROW_7 | ROW_8,
            me: PlayerInfo {
                id: WHITE,
                left_shift: 7,
                fwd_shift: 8,
                right_shift: 9,
                pieces_left: 16,
                won: false,
                ty: ROW_8,
            },
            op: PlayerInfo {
                id: BLACK,
                left_shift: 55,
                fwd_shift: 56,
                right_shift: 57,
                pieces_left: 16,
                won: false,
                ty: ROW_1,
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
            0.0 // TODO test out -1.0
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

        let num_fwd_acs = fwd_to.count_ones();
        let num_right_acs = right_to.count_ones();
        let num_left_acs = left_to.count_ones();

        let i = rng.gen_range(0, num_fwd_acs + num_right_acs + num_left_acs);

        if i >= num_fwd_acs + num_right_acs {
            // generate a left action
            let left_from = left_to.rotate_right(self.me.left_shift as u32);
            ActionIterator(left_from, left_to)
                .nth((i - (num_fwd_acs + num_right_acs)) as usize)
                .unwrap()
        // ActionIterator(left_from, left_to).fast_nth(i - num_fwd_acs - num_right_acs)
        } else if i >= num_fwd_acs {
            // generate a right action
            let right_from = right_to.rotate_right(self.me.right_shift as u32);
            ActionIterator(right_from, right_to)
                .nth((i - num_fwd_acs) as usize)
                .unwrap()
        // ActionIterator(right_from, right_to).fast_nth(i - num_fwd_acs)
        } else {
            // generate a forward action
            let fwd_from = fwd_to.rotate_right(self.me.fwd_shift as u32);
            ActionIterator(fwd_from, fwd_to).nth(i as usize).unwrap()
            // ActionIterator(fwd_from, fwd_to).fast_nth(i)
        }
    }

    fn step(&mut self, action: &Self::Action) -> bool {
        let &(from_sq, to_sq) = action;

        let to = 1 << to_sq;
        let from = 1 << from_sq;

        self.op.pieces_left -= ((self.op_bb >> to_sq) & 1) as u8;
        self.op_bb &= !to;
        self.my_bb = (self.my_bb | to) & !from;

        // note: count_ones() here is slower than keeping track ourselves
        self.me.won = self.me.ty & to != 0 || self.op.pieces_left == 0;

        std::mem::swap(&mut self.me, &mut self.op);
        std::mem::swap(&mut self.my_bb, &mut self.op_bb);

        self.op.won
    }
}
