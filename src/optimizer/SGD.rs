use crate::tensor::Tensor;
use crate::optimizer::Optimizer;

#[derive(Debug)]
pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f32,
}

impl Optimizer for SGD {
    fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self { params, lr }
    }
    
    fn step(&mut self) {
        for param in &mut self.params {
            let grad = param.0.borrow().grad.clone().expect("Gradient not found");
            param.0.borrow_mut().data -= &((self.lr) * grad);
        }
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params {
            param.0.borrow_mut().grad = None;
        }
    }
}