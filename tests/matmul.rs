use ndarray::array;
use torch_rs::{ops::matmul::matmul, tensor::Tensor};
#[test]
fn test_tensor_matmul_2d() {
    let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
    let b = Tensor::new(array![[5.0, 6.0], [7.0, 8.0]].into_dyn());
    let result = matmul(&a, &b);
    // 预期结果:
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    // 结果应该是 3D 的，因为 matmul 默认 b 是带批次的
    let expected = array![[[19.0, 22.0], [43.0, 50.0]]].into_dyn();
    assert_eq!(result.0.borrow().data, expected);
}
#[test]
fn test_tensor_matmul_2d_grad() {
    let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn()).requires_grad(true);
    let b = Tensor::new(array![[5.0, 6.0], [7.0, 8.0]].into_dyn()).requires_grad(true);
    let result = matmul(&a, &b);
    result.backward();
    let expected_grad_a = array![[11.0, 15.0], [11.0, 15.0]].into_dyn();
    assert_eq!(a.0.borrow().grad.clone().unwrap(), expected_grad_a);
    let expected_grad_b = array![[4.0, 4.0], [6.0, 6.0]].into_dyn();
    assert_eq!(b.0.borrow().grad.clone().unwrap(), expected_grad_b);
}
#[test]
fn test_tensor_batch_matmul() {
    // a: [2, 3], b: [2, 3, 4] (batch=2)
    let a = Tensor::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
    let b = Tensor::new(
        array![
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]
        ]
        .into_dyn(),
    );
    let result = matmul(&a, &b);
    // 预期结果形状: [2, 2, 4]
    assert_eq!(result.shape(), vec![2, 2, 4]);
    let expected = array![
        [[38.0, 44.0, 50.0, 56.0], [83.0, 98.0, 113.0, 128.0]],
        [[110.0, 116.0, 122.0, 128.0], [263.0, 278.0, 293.0, 308.0]]
    ]
    .into_dyn();
    assert_eq!(result.0.borrow().data, expected);
}
#[test]
fn test_tensor_batch_matmul_grad() {
    let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn()).requires_grad(true);
    let b = Tensor::new(
        array![[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]].into_dyn(),
    )
    .requires_grad(true);
    let result = matmul(&a, &b);
    result.backward();
    let expected_grad_a = array![[30.0, 38.0], [30.0, 38.0]].into_dyn();
    assert_eq!(a.0.borrow().grad.clone().unwrap(), expected_grad_a);
    let expected_grad_b =
        array![[[4.0, 4.0], [6.0, 6.0]], [[4.0, 4.0], [6.0, 6.0]]].into_dyn();
    assert_eq!(b.0.borrow().grad.clone().unwrap(), expected_grad_b);
}