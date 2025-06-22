use crate::optimizer::Optimizer;
use crate::tensor::Tensor;

/// 随机梯度下降（SGD）优化器。
#[derive(Debug)]
pub struct SGD {
    /// 需要优化的参数列表
    pub params: Vec<Tensor>,
    /// 学习率
    pub lr: f32,
}

impl SGD {
    /// 创建SGD优化器
    ///
    /// # 参数
    /// * `params` - 需要优化的参数
    /// * `lr` - 学习率
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        SGD { params, lr }
    }
    /// 设置学习率
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Optimizer trait的实现，支持参数更新与梯度清零
impl Optimizer for SGD {
    /// 执行一步参数更新
    fn step(&mut self) {
        for param in &mut self.params {
            let grad = param.0.borrow().grad.clone().unwrap_or_else(|| {
                panic!("Gradient not found! param: {:?}", param);
            });
            assert!(
                grad.shape() == param.0.borrow().data.shape(),
                "Gradient shape mismatch! {:?}",
                param
            );
            param.0.borrow_mut().data -= &((self.lr) * grad);
        }
    }

    /// 清零所有参数的梯度
    fn zero_grad(&mut self) {
        for param in &mut self.params {
            param.0.borrow_mut().grad = None;
        }
    }
}
