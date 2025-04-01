// src/autograd/mod.rs
pub mod op; // 声明子模块
use ndarray::ArrayD;
pub use op::{AddOp, Op}; // 导出需要公开的类型

use super::tensor::Tensor;
use std::sync::Arc;

pub fn add(a: &Arc<Tensor>, b: &Arc<Tensor>) -> Tensor {
    let op = Arc::new(AddOp);
    let mut result = op.forward(&[a, b]);

    // 记录计算图关系
    result.add_parent(a);
    result.add_parent(b);
    result.set_creator(op.clone());

    result
}

impl Tensor {
    pub fn backward(&mut self) {
        if !self.requires_grad {
            return;
        }
        *self.grad.write().unwrap() = Some(ArrayD::ones(self.data.shape()));

        let mut grads = vec![self.clone()];
        let mut visited = std::collections::HashSet::new();

        while let Some(tensor) = grads.pop() {
            if visited.contains(&Arc::as_ptr(&tensor.data)) {
                continue;
            }
            visited.insert(Arc::as_ptr(&tensor.data));

            if let Some(ref creator) = tensor.creator {
                creator.backward((*tensor.data).clone());

                for parent in &tensor.parents {
                    if let Some(parent) = parent.upgrade() {
                        {
                            println!("parents{:?}", parent.grad.read().unwrap());
                            let mut parent_grad = parent.grad.write().unwrap();
                            if let Some(ref mut parent_grad) = *parent_grad {
                                *parent_grad += tensor.grad.read().unwrap().as_ref().unwrap();
                            } else {
                                *parent_grad =
                                    Some(tensor.grad.read().unwrap().as_ref().unwrap().clone());
                            }
                        }
                        grads.push((*parent).clone());
                    }
                }
            }
        }
    }
}
