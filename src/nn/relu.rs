use crate::nn::Module;
use crate::tensor::Tensor;
use crate::F;

#[derive(Debug)]
pub struct ReLU;
impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}
impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        F::relu(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn eval(&mut self) {
        // ReLU does not have any specific evaluation behavior
    }
    fn train(&mut self) {
        // ReLU does not have any specific training behavior
    }
}
