use crate::tensor::Tensor;
use std::ops::{Add, Mul};

/// 计算预测值和目标值之间的均方误差 (Mean Squared Error)。
///
/// MSE Loss = mean((prediction - target)^2)
///
/// # 参数
/// * `prediction` - 模型的预测输出张量。
/// * `target` - 真实的标签张量。
///
/// # 返回
/// 一个包含损失值的标量张量。
///
pub fn mse_loss(prediction: &Tensor, target: &Tensor) -> Tensor {
    let diff = prediction + &((-1.0 as f32) * target);
    let squared_diff = &diff * &diff;
    // println!("Squared Difference: {:?}", squared_diff);
    squared_diff.mean()
}

pub fn relu(input: &Tensor) -> Tensor {
    let data = input.0.borrow().data.mapv(|x| x.max(0.0)).into_dyn();
    Tensor::new(data).requires_grad(input.0.borrow().requires_grad)
}
