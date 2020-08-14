extern crate rand;

use crate::rand::rngs::StdRng;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

pub trait Env {
    type Action: Eq + Clone + Copy + std::fmt::Debug;

    fn new() -> Self;
    fn turn(&self) -> usize;
    fn is_over(&self) -> bool;
    fn reward(&self, color: usize) -> f32;
    fn actions(&self) -> Vec<Self::Action>;
    fn get_random_action(&self, rng: &mut StdRng) -> Self::Action;
    fn step(&mut self, action: &Self::Action) -> bool;
}
