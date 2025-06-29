use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::fmt::Debug;
use std::sync::Arc;

/// 自动微分操作trait。
///
/// 需实现前向传播和反向传播。
pub trait Op: Send + Sync + Debug {
    /// 前向传播
    fn forward(&self, inputs: &[&Arc<Tensor>]) -> Tensor;
    /// 反向传播，返回每个输入的梯度
    fn backward(&self, grad: ArrayD<f32>) -> Vec<ArrayD<f32>>;
}

/// 加法操作
#[derive(Debug)]
pub struct AddOp;

impl Op for AddOp {
    fn forward(&self, inputs: &[&Arc<Tensor>]) -> Tensor {
        let a = inputs[0].data.as_ref();
        let b = inputs[1].data.as_ref();
        let result_data = a + b;

        let mut result = Tensor::new(result_data);
        result.requires_grad = inputs[0].requires_grad || inputs[1].requires_grad;
        result
    }
    fn backward(&self, grad: ArrayD<f32>) -> Vec<ArrayD<f32>> {
        vec![grad.clone(), grad.clone()]
    }
}
