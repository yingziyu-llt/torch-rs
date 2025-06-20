use super::Module;
use crate::ops::matmul::matmul;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,
    pub in_features: usize,
    pub out_features: usize,
    pub training: bool,
}

impl Module for Linear {
    fn new(in_features: usize, out_features: usize) -> Self {
        let w = Tensor::randn(&[out_features, in_features]);
        let b = Tensor::zeros(&[out_features]);
        Self {
            w,
            b,
            in_features,
            out_features,
            training: true,
        }
    }
    fn forward(&self, x: &Tensor) -> Tensor {
        &matmul(&self.w,x) + &self.b
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.w.clone(), self.b.clone()]
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
}
