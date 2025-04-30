use super::Op;
use crate::tensor::Tensor;
use ndarray::{ArrayD, Ix2};
use std::rc::Rc;

#[derive(Debug)]
pub struct MatMul {
    input_shapes: Vec<Vec<usize>>,
    // 新增字段存储前向传播时的输入值
    a_data: Option<ArrayD<f32>>,
    b_data: Option<ArrayD<f32>>,
}

impl MatMul {
    pub fn new() -> Self {
        MatMul {
            input_shapes: Vec::new(),
            a_data: None,
            b_data: None,
        }
    }
}

impl Op for MatMul {
    fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        assert_eq!(inputs.len(), 2, "MatMul requires exactly 2 input tensors");

        let a = inputs[0];
        let b = inputs[1];

        // 获取并保存输入数据
        let a_data = a.0.borrow().data.clone();
        let b_data = b.0.borrow().data.clone();

        // 转换为二维数组
        let a_2d = a_data
            .into_dimensionality::<Ix2>()
            .expect("Input A must be convertible to 2D");
        let b_2d = b_data
            .into_dimensionality::<Ix2>()
            .expect("Input B must be convertible to 2D");

        // 检查矩阵维度
        assert_eq!(
            a_2d.shape()[1],
            b_2d.shape()[0],
            "Matrix dimensions mismatch for multiplication"
        );

        // 执行矩阵乘法
        let result = a_2d.dot(&b_2d).into_dyn();

        // 创建输出张量
        let output = Tensor::new(result);

        // 设置计算图关系并保存输入数据
        {
            let mut output_data = output.0.borrow_mut();
            output_data.set_creator(Rc::new(MatMul {
                input_shapes: vec![a_2d.shape().to_vec(), b_2d.shape().to_vec()],
                a_data: Some(a.0.borrow().data.clone()),
                b_data: Some(b.0.borrow().data.clone()),
            }));
            output_data.add_parent(&inputs[0]);
            output_data.add_parent(&inputs[1]);
        }

        output
    }

    fn backward(&self, grad: &Tensor) -> Vec<ArrayD<f32>> {
        // 获取梯度数据
        let grad_data = grad
            .0
            .borrow()
            .grad
            .as_ref()
            .expect("Gradient not set")
            .clone();
        let grad_2d = grad_data.into_dimensionality::<Ix2>().unwrap();

        // 获取前向传播时保存的输入数据
        let a_data = self
            .a_data
            .as_ref()
            .expect("Forward pass data not saved")
            .clone();
        let b_data = self
            .b_data
            .as_ref()
            .expect("Forward pass data not saved")
            .clone();

        let a_2d = a_data.into_dimensionality::<Ix2>().unwrap();
        let b_2d = b_data.into_dimensionality::<Ix2>().unwrap();

        // 正确计算梯度
        // ∂L/∂A = ∂L/∂C · Bᵀ
        let grad_a = grad_2d.dot(&b_2d.t()).into_dyn();

        // ∂L/∂B = Aᵀ · ∂L/∂C
        let grad_b = a_2d.t().dot(&grad_2d).into_dyn();

        vec![grad_a, grad_b]
    }
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let opt = MatMul::new();
    opt.forward(&[a, b])
}
