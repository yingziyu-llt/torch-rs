use crate::optimizer::Optimizer;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f32,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        println!("Using SGD optimizer with learning rate: {}", lr);
        println!("Parameters: {:?}", params);
        SGD { params, lr }
    }
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for param in &mut self.params {
            let grad = param.0.borrow().grad.clone().expect("Gradient not found");
            assert!(
                grad.shape() == param.0.borrow().data.shape(),
                "Gradient shape mismatch! {:?}",
                param
            );
            param.0.borrow_mut().data -= &((self.lr) * grad);
        }
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params {
            param.0.borrow_mut().grad = None;
        }
    }
}
