use std::rc::Rc;

use crate::ops::Op;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct ReLU;
impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}
impl Op for ReLU {
    fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        assert!(inputs.len() == 1, "ReLU expects exactly one input tensor");
        let input = inputs[0];
        let res = Tensor::new(input.0.borrow().data.mapv(|x| x.max(0.0)).into_dyn())
            .require_grad(input.0.borrow().requires_grad);
        let op = ReLU::new();
        res.0.borrow_mut().set_creator(Rc::new(op));
        res.0.borrow_mut().add_parent(input);
        res

    }

    fn backward(&self, parent: &Tensor) -> Vec<ndarray::ArrayD<f32>> {
        let grad = parent.0.borrow().grad.clone().expect("Gradient not found");
        let data = parent.0.borrow().data.clone();
        vec![data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * grad]
    }
}
