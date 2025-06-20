use super::Op;
use crate::tensor::Tensor;
use ndarray::{ArrayD, Ix2, Axis, IxDyn};
use core::panic;
use std::rc::Rc;

#[derive(Debug)]
pub struct MatMul {
    input_shapes: Vec<Vec<usize>>,
    a_data: Option<ArrayD<f32>>,
    b_data: Option<ArrayD<f32>>,
}

impl MatMul {
    pub fn new() -> Self {
        MatMul {
            input_shapes: Vec::new(),
            a_data: None,
            b_data: None,
        }
    }
}

impl Op for MatMul {
    fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        // 暂时先就实现一个简单的矩阵乘法，AxB,A:[output,input],B[batch,input,dim]/[input,dim]
        if inputs.len() != 2 {
            panic!("MatMul requires exactly two input tensors");
        }
        let a = &inputs[0].0.borrow().data;
        let b = &inputs[1].0.borrow().data;
        
        if a.ndim() != 2 {
            panic!("Input A must be a 2D tensor, got {}D", a.ndim());
        }
        let b_3d = if b.ndim() == 2 {
            b.clone().insert_axis(ndarray::Axis(0))
        } else if b.ndim() == 3 {
            b.clone()
        } else {
            panic!("Input B must be a 2D or 3D tensor, got {}D", b.ndim());
        };

        if a.shape()[1] != b_3d.shape()[1] {
            panic!("Matrix dimensions do not match for multiplication: {} and {}", a.shape()[1], b_3d.shape()[1]);
        }

        let a_2d = a.clone().into_dimensionality::<Ix2>().unwrap();
        let mut results = Vec::with_capacity(b_3d.shape()[0]);
        for b_slice in b_3d.axis_iter(Axis(0)) {
            let b_slice = b_slice.into_dimensionality::<Ix2>().unwrap();
            let result = a_2d.dot(&b_slice).into_dyn();
            results.push(result);
        }
        let output = ArrayD::from_shape_vec(
            IxDyn(&[b_3d.shape()[0], a_2d.shape()[0], b_3d.shape()[2]]),
            results.into_iter().flatten().collect(),
        ).unwrap();
        let result = Tensor::new(output);
        if inputs[0].0.borrow().requires_grad || inputs[1].0.borrow().requires_grad {
            result.0.borrow_mut().grad = Some(ArrayD::zeros(result.shape()));
        }
        if inputs[0].0.borrow().requires_grad {
            result.0.borrow_mut().add_parent(inputs[0]);
        }
        if inputs[1].0.borrow().requires_grad {
            result.0.borrow_mut().add_parent(inputs[1]);
        }
        let op = MatMul {
            input_shapes: vec![inputs[0].shape().to_vec(), inputs[1].shape().to_vec()],
            a_data: Some(a.clone()),
            b_data: Some(b.clone()),
        };
        result.0.borrow_mut().set_creator(Rc::new(op));
        result
    }


    fn backward(&self, grad: &Tensor) -> Vec<ArrayD<f32>> {
        // 从传入张量的 .grad 字段获取梯度，而不是 .data 字段
        let grad_output = grad.0.borrow().grad.as_ref().expect("Gradient not found in backward pass").clone();
        let a = self.a_data.as_ref().expect("a_data not saved in MatMul");
        let b = self.b_data.as_ref().expect("b_data not saved in MatMul");

        // 将 b 和 grad_output 统一为 3D 张量以便计算
        let b_3d = if b.ndim() == 2 {
            b.clone().insert_axis(Axis(0))
        } else {
            b.clone()
        };
        let grad_output_3d = if grad_output.ndim() == 2 {
            grad_output.insert_axis(Axis(0))
        } else {
            grad_output
        };

        // 将 a 转换为 2D 视图
        let a_2d = a.view().into_dimensionality::<Ix2>().unwrap();

        // --- 计算 grad_a = sum(grad_output @ b.T, axis=0) ---
        // grad_output_3d: [batch, out_features, dim]
        // b_3d:           [batch, in_features, dim]
        // b_slice.T:      [dim, in_features]
        // result_slice:   [out_features, in_features]
        let mut grad_a = ArrayD::zeros(a.shape());
        let mut grad_a_2d = grad_a.view_mut().into_dimensionality::<Ix2>().unwrap();

        for (grad_slice, b_slice) in grad_output_3d.axis_iter(Axis(0)).zip(b_3d.axis_iter(Axis(0))) {
            let grad_slice_2d = grad_slice.into_dimensionality::<Ix2>().unwrap();
            let b_slice_2d = b_slice.into_dimensionality::<Ix2>().unwrap();
            // 累加每个批次的梯度
            grad_a_2d.scaled_add(1.0, &grad_slice_2d.dot(&b_slice_2d.t()));
        }

        // --- 计算 grad_b = a.T @ grad_output ---
        // a.T:            [in_features, out_features]
        // grad_output_3d: [batch, out_features, dim]
        // result_slice:   [in_features, dim]
        let a_t = a_2d.t();
        let mut grad_b_slices = Vec::new();
        for grad_slice in grad_output_3d.axis_iter(Axis(0)) {
            let grad_slice_2d = grad_slice.into_dimensionality::<Ix2>().unwrap();
            let grad_b_slice = a_t.dot(&grad_slice_2d);
            grad_b_slices.push(grad_b_slice.into_dyn());
        }

        // 将各个批次的梯度切片堆叠成一个完整的 grad_b 张量
        let mut grad_b = ndarray::stack(
            Axis(0),
            &grad_b_slices.iter().map(|arr| arr.view()).collect::<Vec<_>>(),
        ).unwrap();

        // 如果原始输入 b 是 2D，则需要将 grad_b 从 3D 降维回 2D
        if self.input_shapes.get(1).map_or(false, |s| s.len() == 2) {
            grad_b = grad_b.remove_axis(Axis(0)).into_dyn();
        }

        vec![grad_a, grad_b]
    }
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let opt = MatMul::new();
    opt.forward(&[a, b])
}
