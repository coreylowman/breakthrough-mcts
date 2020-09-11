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
        let op_winners =
            self.op_bb & ((ROW_1 << self.op.ty_shift).rotate_right(self.op.fwd_shift as u32));

        let fwd_to = self.my_bb.rotate_left(self.me.fwd_shift as u32) & !self.my_bb & !self.op_bb;
        let right_to =
            (self.my_bb & NOT_COL_8).rotate_left(self.me.right_shift as u32) & !self.my_bb;
        let left_to = (self.my_bb & NOT_COL_1).rotate_left(self.me.left_shift as u32) & !self.my_bb;

        let fwd_win = fwd_to & (ROW_1 << self.me.ty_shift);
        let right_win = right_to & (ROW_1 << self.me.ty_shift);
        let left_win = left_to & (ROW_1 << self.me.ty_shift);

        let fwd_not_lose = fwd_to & op_winners;
        let right_not_lose = right_to & op_winners;
        let left_not_lose = left_to & op_winners;

        if fwd_win != 0 || right_win != 0 || left_win != 0 {
            (fwd_win, right_win, left_win)
        } else if fwd_not_lose != 0 || right_not_lose != 0 || left_not_lose != 0 {
            (fwd_not_lose, right_not_lose, left_not_lose)
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

    fn symmetry_of(action: &Self::Action) -> Self::Action {
        let (from, to) = action;
        (from + 7 - 2 * (from % 8), to + 7 - 2 * (to % 8))
    }

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
            -1.0
        }
    }

    fn num_actions(&self) -> u8 {
        let (fwd_to, right_to, left_to) = self.action_bitboards();

        let num_fwd_acs = fwd_to.count_ones() as u8;
        let num_right_acs = right_to.count_ones() as u8;
        let num_left_acs = left_to.count_ones() as u8;

        num_fwd_acs + num_right_acs + num_left_acs
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
        // assert!(self.actions().contains(action));

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
