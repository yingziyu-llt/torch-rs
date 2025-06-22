use super::Module;
use crate::ops::matmul::matmul;
use crate::tensor::Tensor;

/// 线性层（全连接层），实现 y = xW + b。
#[derive(Debug)]
pub struct Linear {
    /// 权重参数，形状为 (in_features, out_features)
    pub w: Tensor,
    /// 偏置参数，形状为 (out_features,)
    pub b: Tensor,
    /// 输入特征数
    pub in_features: usize,
    /// 输出特征数
    pub out_features: usize,
    /// 是否处于训练模式
    pub training: bool,
}

impl Linear {
    /// 创建一个线性层
    ///
    /// # 参数
    /// * `in_features` - 输入特征数
    /// * `out_features` - 输出特征数
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let w = Tensor::randn(vec![in_features, out_features].as_slice()).require_grad(true);
        let b = Tensor::zeros(vec![out_features].as_slice()).require_grad(true);
        Linear {
            w,
            b,
            in_features,
            out_features,
            training: true,
        }
    }
}

/// Module trait的实现，支持前向传播、参数获取、训练/评估模式切换
impl Module for Linear {
    /// 前向传播，x @ w + b
    fn forward(&self, x: &Tensor) -> Tensor {
        let mid = matmul(x, &self.w);
        &mid + &self.b
    }
    /// 获取所有可训练参数
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.w.clone(), self.b.clone()]
    }
    /// 切换到训练模式
    fn train(&mut self) {
        self.training = true;
    }
    /// 切换到评估模式
    fn eval(&mut self) {
        self.training = false;
    }
}
