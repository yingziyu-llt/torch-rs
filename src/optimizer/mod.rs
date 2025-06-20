#[allow(non_snake_case)]
pub mod SGD;
pub trait Optimizer {
    fn new(params: Vec<crate::tensor::Tensor>, lr: f32) -> Self;
    fn step(&mut self);
    fn zero_grad(&mut self);
}