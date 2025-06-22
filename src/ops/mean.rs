use super::Op;
use crate::tensor::Tensor;
use ndarray::{Array, ArrayD, IxDyn};
use std::rc::Rc;

#[derive(Debug)]
pub struct Mean {
    /// The shape of the input tensor, saved for the backward pass.
    input_shape: Vec<usize>,
}

impl Mean {
    pub fn new(input_shape: Vec<usize>) -> Self {
        Mean { input_shape }
    }
}

impl Op for Mean {
    /// Computes the mean of a tensor.
    fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        if inputs.len() != 1 {
            panic!("Mean operation takes exactly one input.");
        }
        let input = &inputs[0];
        let data = &input.0.borrow().data;

        // Calculate the mean. ndarray's mean() returns Option<f32>.
        let mean_val = data.mean().expect("Cannot compute mean of an empty tensor.");

        // Create a true scalar (0-dim) tensor
        let result_data = Array::from_elem(IxDyn(&[]), mean_val).into_dyn();
        let result = Tensor::new(result_data);

        // Set up the computation graph for backpropagation.
        if input.0.borrow().requires_grad {
            // For a scalar tensor, we need a scalar gradient
            result.0.borrow_mut().grad = Some(ArrayD::from_elem(IxDyn(&[]), 1.0));
            let op = Mean::new(input.shape());
            result.0.borrow_mut().set_creator(Rc::new(op));
            result.0.borrow_mut().add_parent(input);
            result.0.borrow_mut().requires_grad = true;
        } else {
            // Explicitly set requires_grad to false if input doesn't require grad
            result.0.borrow_mut().requires_grad = false;
        }
        result
    }

    /// Computes the gradient of the mean operation.
    fn backward(&self, parent: &Tensor) -> Vec<ArrayD<f32>> {
        // The gradient from the output tensor (which is the parent in the graph)
        let grad_output = parent.0.borrow().grad.as_ref().expect("Gradient not found in backward pass").clone();
        
        let grad_output_scalar = grad_output.iter().next().cloned().unwrap_or(1.0);

        // The number of elements in the original input tensor.
        let num_elements = self.input_shape.iter().product::<usize>() as f32;
        if num_elements == 0.0 {
            return vec![ArrayD::zeros(IxDyn(&self.input_shape))];
        }

        // The gradient for each element of the input is grad_output / N.
        let grad_val = grad_output_scalar / num_elements;

        // Create the gradient array for the input, with the same shape as the original input.
        let grad_input = ArrayD::from_elem(IxDyn(&self.input_shape), grad_val);

        vec![grad_input]
    }
}

/// Functional interface for the mean operation.
pub fn mean(tensor: &Tensor) -> Tensor {
    let op = Mean::new(tensor.shape());
    op.forward(&[tensor])
}

impl Tensor {
    /// Computes the mean of the tensor.
    pub fn mean(&self) -> Tensor {
        mean(self)
    }
}