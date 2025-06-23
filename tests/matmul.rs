use ndarray::array;
use torch_rs::{ops::matmul::matmul, tensor::Tensor};
#[test]
fn test_tensor_matmul_2d() {
    let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
    let b = Tensor::new(array![[5.0, 6.0], [7.0, 8.0]].into_dyn());
    let result = matmul(&a, &b);
    // 预期结果: A:[2,2], B:[2,2] -> Out:[2,2]
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let expected = array![[19.0, 22.0], [43.0, 50.0]].into_dyn();
    assert_eq!(result.0.borrow().data, expected);
}
#[test]
fn test_tensor_matmul_2d_grad() {
    let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn()).require_grad(true);
    let b = Tensor::new(array![[5.0, 6.0], [7.0, 8.0]].into_dyn()).require_grad(true);
    let result = matmul(&a, &b);
    result.backward();
    // grad_a = grad_output @ b.T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
    let expected_grad_a = array![[11.0, 15.0], [11.0, 15.0]].into_dyn();
    assert_eq!(a.0.borrow().grad.clone().unwrap(), expected_grad_a);
    // grad_b = a.T @ grad_output = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    let expected_grad_b = array![[4.0, 4.0], [6.0, 6.0]].into_dyn();
    assert_eq!(b.0.borrow().grad.clone().unwrap(), expected_grad_b);
}
#[test]
fn test_tensor_3d_matmul() {
    // a: [batch, channel, input] -> [2, 2, 3]
    // b: [input, output] -> [3, 4]
    let a = Tensor::new(
        array![
            [[1., 2., 3.], [4., 5., 6.]],
            [[7., 8., 9.], [10., 11., 12.]]
        ]
        .into_dyn(),
    );
    let b = Tensor::new(array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]].into_dyn());
    let result = matmul(&a, &b);
    // 预期结果形状: [2, 2, 4]
    assert_eq!(result.shape(), vec![2, 2, 4]);
    let expected = array![
        [[38.0, 44.0, 50.0, 56.0], [83.0, 98.0, 113.0, 128.0]],
        [[128.0, 152.0, 176.0, 200.0], [173.0, 206.0, 239.0, 272.0]]
    ]
    .into_dyn();
    assert_eq!(result.0.borrow().data, expected);
}
#[test]
fn test_tensor_3d_matmul_grad() {
    // a: [batch, channel, input] -> [1, 2, 2]
    // b: [input, output] -> [2, 2]
    let a = Tensor::new(array![[[1.0, 2.0], [3.0, 4.0]]].into_dyn()).require_grad(true);
    let b = Tensor::new(array![[5.0, 6.0], [7.0, 8.0]].into_dyn()).require_grad(true);
    let result = matmul(&a, &b);
    result.backward();

    // grad_a = grad_output @ b.T
    let expected_grad_a = array![[[11.0, 15.0], [11.0, 15.0]]].into_dyn();
    assert_eq!(a.0.borrow().grad.clone().unwrap(), expected_grad_a);

    // grad_b = a_reshaped.T @ grad_output_reshaped
    let expected_grad_b = array![[4.0, 4.0], [6.0, 6.0]].into_dyn();
    assert_eq!(b.0.borrow().grad.clone().unwrap(), expected_grad_b);
}

#[test]
fn test_vectors_matmul() {
    let a = Tensor::new(array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]].into_dyn()).require_grad(true);
    let b = Tensor::randn(&[3, 4]).require_grad(true);
    let c = Tensor::randn(&[2, 4]).require_grad(true);
    let result = &matmul(&a, &b) + &c;
    // 预期结果: A:[1,3], B:[3,4]
    // Out:[1,4]
    let expected_shape = vec![2, 4];
    assert_eq!(result.shape(), expected_shape);
    result.backward();
    println!("a:{:?}, b:{:?}, c:{:?}", a, b, c);
}
