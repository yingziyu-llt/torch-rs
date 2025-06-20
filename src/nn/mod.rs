pub mod linear;

use crate::tensor::Tensor;
use std::fmt::Debug;

pub trait Module: Debug {
    fn new(in_features: usize, out_features: usize) -> Self;
    /// 前向传播
    fn forward(&self, inputs: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
}
