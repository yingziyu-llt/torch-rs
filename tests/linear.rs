use torch_rs::nn::linear::Linear;
use torch_rs::F::mse_loss;

#[cfg(test)]
mod tests {
    use std::{sync::Arc, vec};

    use torch_rs::nn::Module;

    use super::*;

    #[test]
    fn test_linear() {
        let linear = Linear::new(10, 5);
        assert_eq!(linear.in_features, 10);
        assert_eq!(linear.out_features, 5);
        assert_eq!(linear.w.shape(), &[10, 5]);
        assert_eq!(linear.b.shape(), &[5]);
        let x = torch_rs::tensor::Tensor::ones(vec![1,10].as_slice());
        let y = linear.forward(&x);
        assert_eq!(y.shape(), &[1,5]);
        let loss = &y +&(-1.0 * &torch_rs::tensor::Tensor::ones(vec![1,5].as_slice()));
        println!("Loss: {:?}", loss);
        loss.backward();
        let params = linear.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[10, 5]);
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
