pub mod linear;
pub mod relu;
pub mod sequential;

use crate::tensor::Tensor;
use std::fmt::Debug;

pub trait Module: Debug {
    fn forward(&self, inputs: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
}
