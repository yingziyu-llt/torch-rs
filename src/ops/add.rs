use super::Op;
use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis, IxDyn};
use std::ops::Add;
use std::rc::Rc;

#[derive(Debug)]
pub struct AddOp {
    input_shapes: Vec<Vec<usize>>, // 存储输入张量的形状
}

impl AddOp {
    pub fn new(shape1: Vec<usize>, shape2: Vec<usize>) -> Self {
        AddOp {
            input_shapes: vec![shape1, shape2],
        }
    }

    // 获取广播后的形状
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
}

impl Op for AddOp {
    /// 前向传播：计算输入张量的加法结果
    fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        if inputs.len() != 2 {
            panic!("AddOp requires exactly two input tensors");
        }

        let a = &inputs[0].0.borrow().data;
        let b = &inputs[1].0.borrow().data;

        // 使用ndarray的广播机制处理张量加法
        let result_data = match (a.broadcast(b.shape()), b.broadcast(a.shape())) {
            (Some(a_broadcast), _) => &a_broadcast + b,
            (_, Some(b_broadcast)) => a + &b_broadcast,
            // 如果两个都不能直接广播，尝试计算共同的广播形状
            _ => {
                let shape1 = a.shape();
                let shape2 = b.shape();
                let output_shape = Self::get_broadcasted_shape(shape1, shape2);

                let a_broadcast = a
                    .broadcast(IxDyn(&output_shape))
                    .expect("Failed to broadcast first tensor");
                let b_broadcast = b
                    .broadcast(IxDyn(&output_shape))
                    .expect("Failed to broadcast second tensor");

                &a_broadcast.to_owned() + &b_broadcast
            }
        };

        // 创建一个新的op实例，存储输入形状供反向传播使用
        let op = AddOp::new(a.shape().to_vec(), b.shape().to_vec());

        // 创建新的张量作为输出
        let output = Tensor::new(result_data);
        {
            let mut output_data = output.0.borrow_mut();
            output_data.set_creator(Rc::new(op));

            // 添加需要梯度的父节点
            if inputs[0].0.borrow().requires_grad {
                output_data.add_parent(inputs[0]);
            }

            if inputs[1].0.borrow().requires_grad {
                output_data.add_parent(inputs[1]);
            }

            // 如果任一输入需要梯度，则输出也需要梯度
            output_data.requires_grad =
                inputs[0].0.borrow().requires_grad || inputs[1].0.borrow().requires_grad;
        }

        output
    }

    /// 反向传播：计算梯度并传播到父节点
    fn backward(&self, parent: &Tensor) -> Vec<ArrayD<f32>> {
        let grad = parent
            .0
            .borrow()
            .grad
            .as_ref()
            .expect("Gradient should not be None")
            .clone();
        let mut grads = Vec::with_capacity(2);
        // 对于每个输入形状，处理梯度的反向广播
        for shape in &self.input_shapes {
            let mut result = grad.clone();
            // 先对多余的维度 sum
            while result.ndim() > shape.len() {
                result = result.sum_axis(Axis(0));
            }
            // 再对 shape=1 的维度 sum，倒序处理 axis
            let result_shape: Vec<usize> = result.shape().to_vec();
            let mut sum_axes = Vec::new();
            for (i, (&s, &g)) in shape.iter().zip(result_shape.iter()).enumerate() {
                if s == 1 && g > 1 {
                    sum_axes.push(i);
                }
            }
            sum_axes.sort_unstable_by(|a, b| b.cmp(a));
            for axis in sum_axes {
                result = result.sum_axis(Axis(axis));
            }
            // 确保形状完全匹配
            if result.shape() != shape.as_slice() {
                result = result.into_shape(IxDyn(shape)).unwrap();
            }
            grads.push(result);
        }
        grads
    }
}

impl<'a, 'b> Add<&'a Tensor> for &'b Tensor {
    type Output = Tensor;

    fn add(self, other: &'a Tensor) -> Tensor {
        // 直接使用输入形状创建op
        let op = AddOp::new(
            self.0.borrow().data.shape().to_vec(),
            other.0.borrow().data.shape().to_vec(),
        );
        op.forward(&[self, other])
    }
}
