use super::tensor::Tensor;
use ndarray::ArrayD;
use std::{collections::HashSet, rc::Rc};

impl Tensor {
    /// Backward propagation for the computational graph.
    pub fn backward(&self) {
        let mut grads = vec![self.clone()];
        let mut visited = HashSet::new();

        {
            let mut grad = self.0.borrow_mut();
            grad.grad = Some(ArrayD::ones(grad.data.shape()));
            println!("Tensor grad: {:?}", grad.grad);
        }

        // println!("Backward propagation started");
        // println!("{:?}", self);

        while let Some(tensor) = grads.pop() {
            let tensor_ptr = Rc::as_ptr(&tensor.0);

            if visited.contains(&tensor_ptr) {
                continue;
            }
            visited.insert(tensor_ptr);

            if let Some(ref creator) = tensor.0.borrow().creator {
                let child_grads = creator.backward(&tensor);

                for (parent_weak, child_grad) in tensor.0.borrow().parents.iter().zip(child_grads) {
                    if let Some(parent) = parent_weak.upgrade() {
                        let mut parent_data = parent.borrow_mut();

                        match parent_data.grad {
                            Some(ref mut parent_grad) => {
                                *parent_grad += &child_grad;
                            }
                            None => {
                                parent_data.grad = Some(child_grad);
                            }
                        }

                        grads.push(Tensor(parent_weak.upgrade().unwrap()));
                    }
                }
            }
        }
    }
}
