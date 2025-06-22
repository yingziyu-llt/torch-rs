#[cfg(test)]
mod tests {
    use ndarray::array;
    use torch_rs::tensor::Tensor;

    #[test]
    fn test_tensor_add() {
        let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
        let b = Tensor::new(array![[5.0, 6.0], [7.0, 8.0]].into_dyn());

        let result = &a + &b;

        let expected = array![[6.0, 8.0], [10.0, 12.0]].into_dyn();
        assert_eq!(result.0.borrow().data, expected);
    }

    #[test]
    fn test_tensor_add_grad() {
        let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn()).requires_grad(true);
        let b = Tensor::new(array![[5.0, 6.0], [7.0, 8.0]].into_dyn()).requires_grad(true);

        let result = &a + &b;
        result.backward();

        let expected = array![[1.0, 1.0], [1.0, 1.0]].into_dyn();
        assert_eq!(a.0.borrow().grad.clone().unwrap(), expected);
        assert_eq!(b.0.borrow().grad.clone().unwrap(), expected);
    }
    
    // 测试多批次张量加法
    #[test]
    fn test_tensor_batch_add() {
        // 创建两个3D张量，第一个维度是批次
        // 形状为 [2, 2, 3]，表示2个批次，每个批次是2x3的矩阵
        let a = Tensor::new(array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ].into_dyn());
        
        let b = Tensor::new(array![
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        ].into_dyn());

        // 执行加法操作
        let result = &a + &b;

        // 验证结果形状和值
        assert_eq!(result.shape(), vec![2, 2, 3]);
        
        let expected = array![
            [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]],
            [[7.7, 8.8, 9.9], [11.0, 12.1, 13.2]]
        ].into_dyn();
        
        assert_eq!(result.0.borrow().data, expected);
    }

    #[test]
    fn test_tensor_batch_add_grad() {
        // 创建两个需要梯度的3D张量，批次大小为2
        let a = Tensor::new(array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ].into_dyn()).requires_grad(true);
        
        let b = Tensor::new(array![
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        ].into_dyn()).requires_grad(true);

        // 执行加法操作
        let result = &a + &b;
        
        // 反向传播
        result.backward();

        // 验证梯度值 - 对于加法，梯度应该是1
        let expected_grad = array![
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ].into_dyn();
        
        assert_eq!(a.0.borrow().grad.clone().unwrap(), expected_grad);
        assert_eq!(b.0.borrow().grad.clone().unwrap(), expected_grad);
    }
    
    #[test]
    fn test_tensor_batch_broadcasting() {
        // 测试批次维度的广播
        // 一个带批次的张量 [3, 2, 2] 和一个无批次的张量 [2, 2]
        let batch_tensor = Tensor::new(array![
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]]
        ].into_dyn());
        
        let single_tensor = Tensor::new(array![[0.1, 0.2], [0.3, 0.4]].into_dyn());
        
        // 单个张量应该被广播到每个批次
        let result = &batch_tensor + &single_tensor;
        
        // 验证结果
        let expected = array![
            [[1.1, 2.2], [3.3, 4.4]],
            [[5.1, 6.2], [7.3, 8.4]],
            [[9.1, 10.2], [11.3, 12.4]]
        ].into_dyn();
        
        assert_eq!(result.shape(), vec![3, 2, 2]);
        assert_eq!(result.0.borrow().data, expected);
    }
    
    #[test]
    fn test_tensor_batch_broadcasting_grad() {
        // 测试带批次张量的梯度广播
        let batch_tensor = Tensor::new(array![
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ].into_dyn()).requires_grad(true);
        
        let single_tensor = Tensor::new(array![[0.1, 0.2], [0.3, 0.4]].into_dyn()).requires_grad(true);
        
        // 执行加法
        let result = &batch_tensor + &single_tensor;
        result.backward();
        
        // 批次张量的梯度应该和输入形状相同
        let batch_expected_grad = array!(
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]]
        ).into_dyn();
        
        // 单个张量的梯度应该是批次维度上的所有梯度之和
        let single_expected_grad = array![[2.0, 2.0], [2.0, 2.0]].into_dyn();
        
        assert_eq!(batch_tensor.0.borrow().grad.clone().unwrap(), batch_expected_grad);
        assert_eq!(single_tensor.0.borrow().grad.clone().unwrap(), single_expected_grad);
    }

    #[test]
    fn test_tensor_add_broadcasting() {
        // 测试广播机制
        let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
        let b = Tensor::new(array![5.0, 6.0].into_dyn()); // 1D 张量

        let result = &a + &b; // b 应该被广播到 a 的形状

        let expected = array![[6.0, 8.0], [8.0, 10.0]].into_dyn();
        assert_eq!(result.0.borrow().data, expected);
    }
}
