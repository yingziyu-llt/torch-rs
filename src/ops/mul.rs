use crate::ops::Op;
use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis, IxDyn};
use std::ops::Mul;
use std::{process::Output, rc::Rc};

#[derive(Debug)]
pub struct Multiply {
    input_shapes: Vec<Vec<usize>>,
    a_data: Option<ndarray::ArrayD<f32>>,
    b_data: Option<ndarray::ArrayD<f32>>,
}

impl Multiply {
    pub fn new(input_shapes: Vec<Vec<usize>>) -> Self {
        Multiply {
            input_shapes,
            a_data: None,
            b_data: None,
        }
    }

    fn get_broadcasted_shape(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
        let len1 = shape1.len();
        let len2 = shape2.len();
        let max_len = std::cmp::max(len1, len2);

        let mut result = Vec::with_capacity(max_len);

        // 从尾部开始，依次比较对齐的维度
        for i in 0..max_len {
            let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
            let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

            if dim1 != 1 && dim2 != 1 && dim1 != dim2 {
                panic!(
                    "Incompatible dimensions for broadcasting: {} and {}",
                    dim1, dim2
                );
            }

            result.push(std::cmp::max(dim1, dim2));
        }

        result.reverse();
        result
    }

    /// 将张量按目标形状进行求和的辅助函数。
    /// 用于广播操作的反向传播。
    fn sum_to_shape(grad: &ArrayD<f32>, target_shape: &[usize]) -> ArrayD<f32> {
        let mut result = grad.clone();
        let grad_shape = grad.shape();

        // 1. 处理维度数不同的情况 (对多余的前导维度求和)
        if target_shape.len() < grad_shape.len() {
            let axes_to_sum: Vec<usize> = (0..(grad_shape.len() - target_shape.len())).collect();
            for &axis in axes_to_sum.iter().rev() {
                result = result.sum_axis(Axis(axis));
            }
        }

        // 2. 处理被广播的维度 (size 1 -> size > 1)
        let mut sum_axes = Vec::new();
        let current_shape = result.shape(); // 获取求和后的新形状
        for i in 0..target_shape.len() {
            // 如果目标维度是1，但当前梯度维度大于1，说明发生了广播
            if target_shape[i] == 1 && current_shape[i] > 1 {
                sum_axes.push(i);
            }
        }

        // 从大到小排序轴，以避免求和后索引变化
        sum_axes.sort_unstable_by(|a, b| b.cmp(a));
        for axis in sum_axes {
            result = result.sum_axis(Axis(axis));
        }

        // 3. 确保形状完全匹配 (恢复被求和掉的 size 1 的维度)
        if result.shape() != target_shape {
            result = result.into_shape(IxDyn(target_shape)).unwrap();
        }

        result
    }
}

impl Op for Multiply {
    fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        // 逐元素相乘，广播机制
        let a = &inputs[0].0.borrow().data;
        let b = &inputs[1].0.borrow().data;

        let result = match (a.broadcast(b.shape()), b.broadcast(a.shape())) {
            (Some(a_broadcast), _) => &a_broadcast * b,
            (_, Some(b_broadcast)) => a * &b_broadcast,
            // 如果两个都不能直接广播，尝试计算共同的广播形状
            _ => {
                let shape1 = a.shape();
                let shape2 = b.shape();
                let broadcast_shape = Self::get_broadcasted_shape(shape1, shape2);
                let a_broadcasted = a.broadcast(IxDyn(&broadcast_shape)).unwrap_or_else(|| {
                    panic!(
                        "Cannot broadcast tensor with shape {:?} to {:?}",
                        shape1, broadcast_shape
                    )
                });
                let b_broadcasted = b.broadcast(IxDyn(&broadcast_shape)).unwrap_or_else(|| {
                    panic!(
                        "Cannot broadcast tensor with shape {:?} to {:?}",
                        shape2, broadcast_shape
                    )
                });
                a_broadcasted.to_owned() * b_broadcasted.to_owned()
            }
        };

        let op = Rc::new(Multiply {
            input_shapes: self.input_shapes.clone(),
            a_data: Some(a.clone()),
            b_data: Some(b.clone()),
        });

        let result_tensor = Tensor::new(result);
        result_tensor.0.borrow_mut().set_creator(op);
        result_tensor.0.borrow_mut().add_parent(&inputs[0]);
        result_tensor.0.borrow_mut().add_parent(&inputs[1]);
        result_tensor.0.borrow_mut().requires_grad =
            inputs[0].0.borrow().requires_grad || inputs[1].0.borrow().requires_grad;
        result_tensor
    }
    fn backward(&self, parent: &Tensor) -> Vec<ArrayD<f32>> {
        let grad_output = parent
            .0
            .borrow()
            .grad
            .as_ref()
            .expect("Parent gradient is None")
            .clone();
        let a_data = self.a_data.as_ref().expect("a_data is None in backward");
        let b_data = self.b_data.as_ref().expect("b_data is None in backward");

        // 相对于 a 的梯度: ∂L/∂a = ∂L/∂c * b
        let mut grad_a = &grad_output * b_data;
        // 如果发生了广播，需要将梯度求和到原始形状
        if grad_a.shape() != a_data.shape() {
            grad_a = Self::sum_to_shape(&grad_a, a_data.shape());
        }

        // 相对于 b 的梯度: ∂L/∂b = ∂L/∂c * a
        let mut grad_b = &grad_output * a_data;
        // 如果发生了广播，需要将梯度求和到原始形状
        if grad_b.shape() != b_data.shape() {
            grad_b = Self::sum_to_shape(&grad_b, b_data.shape());
        }

        vec![grad_a, grad_b]
    }
}

impl<'a, 'b> Mul<&'a Tensor> for &'b Tensor {
    type Output = Tensor;

    fn mul(self, other: &'a Tensor) -> Tensor {
        let op = Multiply::new(vec![self.shape().to_vec(), other.shape().to_vec()]);
        let inputs = vec![self, other];
        op.forward(&inputs)
    }
}

impl<'a> Mul<f32> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Tensor {
        let b: Tensor = Tensor::from(vec![scalar]);
        self * &b
    }
}

impl<'a> Mul<&'a Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, tensor: &'a Tensor) -> Tensor {
        tensor * self
    }
}

impl<'a> Mul<f64> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f64) -> Tensor {
        let b: Tensor = Tensor::from(vec![scalar as f32]);
        self * &b
    }
}

impl<'a> Mul<&'a Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, tensor: &'a Tensor) -> Tensor {
        tensor * self as f32
    }
}
