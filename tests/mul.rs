use torch_rs::tensor::Tensor;

#[test]
fn test_mul() {
    // 创建两个张量
    let a = Tensor::from(vec![1.0, 2.0, 3.0]).reshape(&[3, 1]).unwrap();
    let b = Tensor::from(vec![4.0, 5.0, 6.0]).reshape(&[3, 1]).unwrap();

    let result = &a * &b;

    let expected = Tensor::from(vec![4.0, 10.0, 18.0])
        .reshape(&[3, 1])
        .unwrap();

    // 验证结果
    assert_eq!(result.0.borrow().data, expected.0.borrow().data);
}
#[test]
fn test_mul_scalar() {
    // 创建一个张量
    let a = Tensor::from(vec![1.0, 2.0, 3.0]).reshape(&[3, 1]).unwrap();

    // 乘以一个标量
    let result: Tensor = &a * 2.0;

    let expected = Tensor::from(vec![2.0, 4.0, 6.0]).reshape(&[3, 1]).unwrap();

    // 验证结果
    assert_eq!(result.0.borrow().data, expected.0.borrow().data);
}

#[test]
fn test_mul_scalar_grad() {
    // 创建一个张量
    let a = Tensor::from(vec![1.0, 2.0, 3.0]).reshape(&[3, 1]).unwrap();

    // 乘以一个标量
    let result: Tensor = &a * 2.0;

    // 计算梯度
    result.backward();

    // 验证梯度
    let grad = a.0.borrow().grad.clone().unwrap();
    let expected_grad = Tensor::from(vec![2.0, 2.0, 2.0]).reshape(&[3, 1]).unwrap();

    assert_eq!(grad, expected_grad.0.borrow().data);
}
