pub mod linear;
pub mod relu;
pub mod sequential;

use crate::tensor::Tensor;
use std::fmt::Debug;

/// 神经网络模块通用trait。
///
/// 支持前向传播、参数获取、训练/评估模式切换。
pub trait Module: Debug {
    /// 前向传播
    fn forward(&self, inputs: &Tensor) -> Tensor;
    /// 获取所有可训练参数
    fn parameters(&self) -> Vec<Tensor>;
    /// 切换到训练模式
    fn train(&mut self);
    /// 切换到评估模式
    fn eval(&mut self);
}
