pub mod add;
pub mod matmul;
pub mod mean;
pub mod mul;
pub mod relu;
use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::fmt::Debug;

pub trait Op: Debug {
    /// 前向传播
    fn forward(&self, inputs: &[&Tensor]) -> Tensor;

    /// 反向传播（返回输入梯度）
    fn backward(&self, parent: &Tensor) -> Vec<ArrayD<f32>>;
}
