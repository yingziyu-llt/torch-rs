use torch_rs::nn::linear::Linear;
use torch_rs::F::mse_loss;

#[cfg(test)]
mod tests {
    use std::{sync::Arc, vec};

    use torch_rs::nn::Module;
    use torch_rs::tensor::Tensor;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_linear() {
        let linear = Linear::new(5, 5);
        assert_eq!(linear.in_features, 5);
        assert_eq!(linear.out_features, 5);
        assert_eq!(linear.w.shape(), &[5, 5]);
        assert_eq!(linear.b.shape(), &[5]);
        let input = Tensor::ones(vec![1, 5].as_slice()).requires_grad(true);
        let y = linear.forward(&input);
        println!("input: {:?}", input);
        assert_eq!(y.shape(), &[1,5]);
        let target = Tensor::new(array![[1.0, 2.0, 3.0, 4.0, 5.0]].into_dyn());
        let loss = mse_loss(&y, &target);
        println!("loss: {:?}", loss);
        loss.backward();
        let params = linear.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[5, 5]);
        assert_eq!(params[1].shape(), &[5]);
        println!("Gradient for w: {:?}", params[0].0.borrow().grad);
        println!("Gradient for b: {:?}", params[1].0.borrow().grad);
    }
    #[test]
    fn test_linear_batch() {
        let linear = Linear::new(10, 5);
        assert_eq!(linear.in_features, 10);
        assert_eq!(linear.out_features, 5);
        assert_eq!(linear.w.shape(), &[10, 5]);
        assert_eq!(linear.b.shape(), &[5]);
        let x = torch_rs::tensor::Tensor::ones(vec![3, 10].as_slice());
        let y = linear.forward(&x);
        assert_eq!(y.shape(), &[3, 5]);
    }
}
