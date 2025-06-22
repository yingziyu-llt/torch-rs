#[allow(non_snake_case)]
pub mod SGD;
pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}