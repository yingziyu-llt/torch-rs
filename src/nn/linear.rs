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
impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let w = Tensor::randn(vec![in_features, out_features].as_slice()).requires_grad(true);
        let b = Tensor::zeros(vec![out_features].as_slice()).requires_grad(true);
        Linear {
            w,
            b,
            in_features,
            out_features,
            training: true,
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        matmul(x,&self.w)
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
