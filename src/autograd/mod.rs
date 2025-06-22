use super::tensor::Tensor;
use ndarray::ArrayD;
use std::{collections::HashSet, rc::Rc};

impl Tensor {
    /// 反向传播：从当前张量出发，递归地计算所有依赖张量的梯度。
    ///
    /// 通常用于loss.backward()。
    pub fn backward(&self) {
        // 1. 初始化自身的梯度为全1（通常用于标量loss）
        {
            let mut self_data = self.0.borrow_mut();
            self_data.grad = Some(ArrayD::ones(self_data.data.shape()));
        }

        // 2. 拓扑排序，确保每个节点在所有子节点之后被处理
        fn topo_sort(tensor: &Tensor, visited: &mut HashSet<*const ()>, order: &mut Vec<Tensor>) {
            let ptr = Rc::as_ptr(&tensor.0) as *const ();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);
            let parents = tensor.0.borrow().parents.clone();
            for parent_weak in parents {
                let parent_rc = parent_weak.clone();
                topo_sort(&Tensor(parent_rc), visited, order);
            }
            order.push(tensor.clone());
        }

        let mut visited = HashSet::new();
        let mut topo_order = Vec::new();
        topo_sort(self, &mut visited, &mut topo_order);

        // 3. 反向遍历拓扑序，执行梯度传播
        for tensor in topo_order.into_iter().rev() {
            let creator = tensor.0.borrow().creator.clone();
            //println!("Processing tensor: {:?}", tensor);
            if let Some(op) = creator {
                let grads = op.backward(&tensor);
                let parents = tensor.0.borrow().parents.clone();
                for (parent_weak, grad) in parents.into_iter().zip(grads.into_iter()) {
                    let parent_rc = parent_weak.clone();
                    let mut parent_data = parent_rc.borrow_mut();
                    match &mut parent_data.grad {
                        Some(parent_grad) => {
                            *parent_grad += &grad;
                        }
                        None => {
                            parent_data.grad = Some(grad);
                        }
                    }
                }
            }
        }
    }
}
