use ndarray::{Array, IxDyn};
use torch_rs::tensor::Tensor;

#[test]
fn test_tensor_creation() {
    // 测试基本创建
    let data = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let tensor = Tensor::new(data.clone());
    assert_eq!(tensor.shape(), vec![2, 3]);
    assert_eq!(tensor.numel(), 6);

    // 测试工厂方法
    let zeros = Tensor::zeros(&[2, 3]);
    assert_eq!(zeros.shape(), vec![2, 3]);
    for &val in zeros.data().iter() {
        assert_eq!(val, 0.0);
    }

    let ones = Tensor::ones(&[2, 3]);
    assert_eq!(ones.shape(), vec![2, 3]);
    for &val in ones.data().iter() {
        assert_eq!(val, 1.0);
    }

    // 测试随机张量
    let rand = Tensor::randn(&[2, 3]);
    assert_eq!(rand.shape(), vec![2, 3]);
    assert_eq!(rand.numel(), 6);

    // 测试 _like 方法
    let tensor = Tensor::ones(&[3, 4]);
    let zeros_like = tensor.zeros_like();
    assert_eq!(zeros_like.shape(), vec![3, 4]);
    for &val in zeros_like.data().iter() {
        assert_eq!(val, 0.0);
    }

    let ones_like = tensor.ones_like();
    assert_eq!(ones_like.shape(), vec![3, 4]);
    for &val in ones_like.data().iter() {
        assert_eq!(val, 1.0);
    }

    let rand_like = tensor.rand_like();
    assert_eq!(rand_like.shape(), vec![3, 4]);
}

#[test]
fn test_tensor_shape_operations() {
    // 测试 view/reshape
    let tensor = Tensor::ones(&[2, 3]);
    let reshaped = tensor.reshape(&[3, 2]).unwrap();
    assert_eq!(reshaped.shape(), vec![3, 2]);

    // 测试view失败情况
    let result = tensor.view(&[4, 4]);
    assert!(result.is_err());

    // 测试 squeeze
    let tensor = Tensor::ones(&[2, 1, 3]);
    let squeezed = tensor.squeeze(Some(1)).unwrap();
    assert_eq!(squeezed.shape(), vec![2, 3]);

    // 测试所有维度的 squeeze
    let tensor = Tensor::ones(&[2, 1, 3, 1]);
    let squeezed = tensor.squeeze(None).unwrap();
    assert_eq!(squeezed.shape(), vec![2, 3]);

    // 测试 squeeze 失败情况
    let tensor = Tensor::ones(&[2, 2, 3]);
    let result = tensor.squeeze(Some(1));
    assert!(result.is_err());

    // 测试 unsqueeze
    let tensor = Tensor::ones(&[2, 3]);
    let unsqueezed = tensor.unsqueeze(1).unwrap();
    assert_eq!(unsqueezed.shape(), vec![2, 1, 3]);

    // 测试 unsqueeze 在末尾
    let tensor = Tensor::ones(&[2, 3]);
    let unsqueezed = tensor.unsqueeze(2).unwrap();
    assert_eq!(unsqueezed.shape(), vec![2, 3, 1]);

    // 测试 unsqueeze 失败情况
    let tensor = Tensor::ones(&[2, 3]);
    let result = tensor.unsqueeze(4);
    assert!(result.is_err());
}

#[test]
fn test_tensor_grad() {
    // 测试 requires_grad 默认值
    let tensor = Tensor::ones(&[2, 3]);
    assert_eq!(tensor.is_leaf(), true); // 新创建的张量应该是叶节点

    // 测试设置 requires_grad
    let tensor = tensor.require_grad(true);
    assert!(tensor.0.borrow().requires_grad);
    assert!(tensor.0.borrow().grad.is_some());

    // 测试取消 requires_grad
    let tensor = tensor.require_grad(false);
    assert!(!tensor.0.borrow().requires_grad);
}

#[test]
fn test_tensor_data_access() {
    // 测试数据访问
    let data = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let tensor = Tensor::new(data.clone());
    assert_eq!(tensor.data(), data);

    // 测试 size 方法
    assert_eq!(tensor.size(0), Some(2));
    assert_eq!(tensor.size(1), Some(3));
    assert_eq!(tensor.size(2), None); // 超出维度应返回None

    // 测试 dim 方法
    assert_eq!(tensor.dim(), 2);

    // 测试 item 方法
    let scalar = Tensor::new(Array::from_shape_vec(IxDyn(&[]), vec![42.0]).unwrap());
    let result = scalar.item();
    assert!(result.is_ok());

    // 测试 item 对非标量的错误情况
    let tensor = Tensor::ones(&[2, 2]);
    let result = tensor.item();
    assert!(result.is_err());
}

#[test]
fn test_tensor_index() {
    // 创建一个3x3张量
    let data = Array::from_shape_vec(
        IxDyn(&[3, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let tensor = Tensor::new(data);

    // 测试索引超出范围
    let result = tensor.index(&[3, 0]);
    assert!(result.is_err());

    // 测试维度不匹配
    let result = tensor.index(&[1]);
    assert!(result.is_err());
}

#[test]
fn test_tensor_detach() {
    // 创建需要梯度的张量
    let tensor = Tensor::ones(&[2, 3]).require_grad(true);

    // 验证初始状态
    assert!(tensor.0.borrow().requires_grad);
    assert!(tensor.is_leaf());

    // 执行detach操作
    let detached = tensor.detach();

    // 验证detached张量的状态
    assert!(!detached.0.borrow().requires_grad);
    assert!(detached.0.borrow().creator.is_none());
    assert_eq!(detached.0.borrow().parents.len(), 0);
    assert!(detached.0.borrow().grad.is_none());

    // 验证原始张量不受影响
    assert!(tensor.0.borrow().requires_grad);
    assert!(tensor.is_leaf());
}

#[test]
fn test_tensor_debug_format() {
    // 测试小张量的格式
    let tensor = Tensor::ones(&[2, 3]);
    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.starts_with("tensor("));
    // assert!(debug_str.contains("[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]"));

    // 测试大张量的格式
    let tensor = Tensor::ones(&[10, 10]);
    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.contains("[...tensor of size 10×10]"));

    // 测试带梯度的格式
    let tensor = Tensor::ones(&[2, 3]).require_grad(true);
    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.contains("requires_grad=true"));
}
