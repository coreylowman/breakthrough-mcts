use crate::env::{Env, BLACK, WHITE};

use crate::rand::prelude::SliceRandom;
use crate::rand::rngs::StdRng;

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
enum Piece {
    Empty = 2,
    White = 0,
    Black = 1,
}

type Board = [[Piece; 8]; 8]; // note: single dimensional vec is less efficient
type Square = u32;

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
struct NaiveEnv {
    player: PlayerInfo,
    opponent: PlayerInfo,
    board: Board,
}

impl Env for NaiveEnv {
    type Action = (Square, Square);

    fn new() -> NaiveEnv {
        let mut board = [[Piece::Empty; 8]; 8];
        for x in 0..8 {
            board[0][x] = Piece::White;
            board[1][x] = Piece::White;
            board[6][x] = Piece::Black;
            board[7][x] = Piece::Black;
        }

        NaiveEnv {
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

    fn turn(&self) -> usize {
        self.player.id
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

    fn actions(&self) -> Vec<Self::Action> {
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

        let winning_actions: Vec<Self::Action> = acs
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

    fn get_random_action(&self, mut rng: &mut StdRng) -> Self::Action {
        let actions = self.actions();
        *actions.choose(&mut rng).unwrap()
    }

    fn step(&mut self, action: &Self::Action) -> bool {
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
