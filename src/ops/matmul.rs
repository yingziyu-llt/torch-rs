use super::Op;
use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis, Ix2, Ix3, IxDyn};
use core::panic;
use std::rc::Rc;
use ndarray_einsum::tensordot;

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
        // A: [batch, input] or [batch, channel, input]
        // B: [input, output]
        // Output: [batch, output] or [batch, channel, output]
        if inputs.len() != 2 {
            panic!("MatMul requires exactly two input tensors");
        }
        let a = &inputs[0].0.borrow().data;
        let b = &inputs[1].0.borrow().data;

        if b.ndim() != 2 {
            panic!("Input B must be a 2D tensor (weights), got {}D", b.ndim());
        }

        let a_ndim = a.ndim();
        if a_ndim != 2 && a_ndim != 3 {
            panic!("Input A must be a 2D or 3D tensor, got {}D", a_ndim);
        }

        let a_last_dim_idx = a_ndim - 1;
        let a_last_dim_size = a.shape()[a_last_dim_idx];
        let b_first_dim_size = b.shape()[0];

        if a_last_dim_size != b_first_dim_size {
            panic!(
                "Matrix dimensions do not match for multiplication. A's last dim ({}) is {} and B's first dim ({}) is {}",
                a_last_dim_idx, a_last_dim_size, 0, b_first_dim_size
            );
        }

        let output_data = if a_ndim == 2 {
            // A: [batch, input], B: [input, output] -> Output: [batch, output]
            let a_2d = a.view().into_dimensionality::<Ix2>().unwrap();
            let b_2d = b.view().into_dimensionality::<Ix2>().unwrap();
            a_2d.dot(&b_2d).into_dyn()
        } else {
            // a_ndim == 3
            // A: [batch, channel, input], B: [input, output] -> Output: [batch, channel, output]
            tensordot(a, b, &[Axis(2)], &[Axis(0)]).into_dyn()
        };

        let result = Tensor::new(output_data);
        if inputs[0].0.borrow().requires_grad || inputs[1].0.borrow().requires_grad {
            result.0.borrow_mut().grad = Some(ArrayD::zeros(result.shape()));
            let op = MatMul {
                input_shapes: vec![inputs[0].shape().to_vec(), inputs[1].shape().to_vec()],
                a_data: Some(a.clone()),
                b_data: Some(b.clone()),
            };
            result.0.borrow_mut().set_creator(Rc::new(op));
        }
        if inputs[0].0.borrow().requires_grad {
            result.0.borrow_mut().add_parent(inputs[0]);
        }
        if inputs[1].0.borrow().requires_grad {
            result.0.borrow_mut().add_parent(inputs[1]);
        }
        result.0.borrow_mut().requires_grad =
            inputs[0].0.borrow().requires_grad || inputs[1].0.borrow().requires_grad;
        result
    }

    fn backward(&self, grad: &Tensor) -> Vec<ArrayD<f32>> {
        let grad_output = grad
            .0
            .borrow()
            .grad
            .as_ref()
            .expect("Gradient not found in backward pass")
            .clone();
        let a = self.a_data.as_ref().expect("a_data not saved in MatMul");
        let b = self.b_data.as_ref().expect("b_data not saved in MatMul");

        let a_ndim = a.ndim();
        let b_2d = b.view().into_dimensionality::<Ix2>().unwrap();

        // --- Calculate grad_a = grad_output @ b.T ---
        let b_t = b_2d.t();
        let grad_a = if a_ndim == 2 {
            let grad_output_2d = grad_output.view().into_dimensionality::<Ix2>().unwrap();
            grad_output_2d.dot(&b_t).into_dyn()
        } else {
            // a_ndim == 3
            // grad_output: [B, C, N], b_t: [N, K] -> grad_a: [B, C, K]
            tensordot(&grad_output, &b_t, &[Axis(2)], &[Axis(0)]).into_dyn()
        };

        // --- Calculate grad_b = a.T @ grad_output ---
        // This involves summing over the batch dimension (and channel if it exists).
        let grad_b = if a_ndim == 2 {
            // a: [B, K], grad_output: [B, N] -> grad_b: [K, N]
            let a_2d = a.view().into_dimensionality::<Ix2>().unwrap();
            let grad_output_2d = grad_output.view().into_dimensionality::<Ix2>().unwrap();
            a_2d.t().dot(&grad_output_2d).into_dyn()
        } else {
            // a_ndim == 3
            // a: [B, C, K], grad_output: [B, C, N] -> grad_b: [K, N]
            // We need to contract the first two axes (batch and channel).
            tensordot(a, &grad_output, &[Axis(0), Axis(1)], &[Axis(0), Axis(1)]).into_dyn()
        };

        vec![grad_a, grad_b]
    }
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let opt = MatMul::new();
    opt.forward(&[a, b])
}
