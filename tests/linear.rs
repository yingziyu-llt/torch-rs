use torch_rs::nn::linear::Linear;

#[cfg(test)]
mod tests {
    use torch_rs::nn::Module;

    use super::*;

    #[test]
    fn test_linear() {
        let linear = Linear::new(10, 5);
        assert_eq!(linear.in_features, 10);
        assert_eq!(linear.out_features, 5);
        assert_eq!(linear.w.shape(), &[5, 10]);
        assert_eq!(linear.b.shape(), &[5]);
        let x = torch_rs::tensor::Tensor::ones(&[10,1]);
        let y = linear.forward(&x);
        assert_eq!(y.shape(), &[5]);
    }
}
