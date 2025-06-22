#[allow(non_snake_case)]
pub mod SGD;
/// 优化器通用trait。
///
/// 支持参数更新与梯度清零。
pub trait Optimizer {
    /// 执行一步参数更新
    fn step(&mut self);
    /// 清零所有参数的梯度
    fn zero_grad(&mut self);
}
