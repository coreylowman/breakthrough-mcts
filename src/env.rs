extern crate rand;

use crate::rand::rngs::StdRng;

pub const WHITE: bool = true;
pub const BLACK: bool = false;

pub trait Env {
    type Action: Eq + Clone + Copy + std::fmt::Debug;
    type ActionIterator: Iterator<Item = Self::Action>;

    fn symmetry_of(action: &Self::Action) -> Self::Action;

    fn new() -> Self;
    fn turn(&self) -> bool;
    fn is_over(&self) -> bool;
    fn reward(&self, color: bool) -> f32;
    fn iter_actions(&self) -> Self::ActionIterator;
    fn actions(&self) -> Vec<Self::Action>;
    fn get_random_action(&self, rng: &mut StdRng) -> Self::Action;
    fn step(&mut self, action: &Self::Action) -> bool;
}
