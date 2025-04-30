use super::Op;
use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::ops::Add;
use std::rc::Rc;

#[derive(Debug)]
pub struct AddOp {
    input_shapes: Vec<Vec<usize>>, // 存储输入张量的形状
}

impl AddOp {
    pub fn new() -> Self {
        AddOp {
            input_shapes: Vec::new(),
        }
    }
}

impl Op for AddOp {
    /// 前向传播：计算输入张量的加法结果
    fn forward(&self, inputs: &[&Tensor]) -> Tensor {
        // 验证输入数量
        if inputs.len() != 2 {
            panic!("AddOp requires exactly two input tensors");
        }

        // 获取输入张量的数据
        let data1 = inputs[0].0.borrow().data.clone();
        let data2 = inputs[1].0.borrow().data.clone();

        // 执行加法操作
        let result_data = &data1 + &data2;

        // 创建新的张量作为输出
        let output = Tensor::new(result_data);
        {
            // 设置创建者为当前操作
            let mut output_data = output.0.borrow_mut();
            output_data.set_creator(Rc::new(AddOp {
                input_shapes: vec![data1.shape().to_vec(), data2.shape().to_vec()],
            }));

            // 添加父节点
            output_data.add_parent(&inputs[0]);
            output_data.add_parent(&inputs[1]);
        }
        output
    }

    /// 反向传播：计算梯度并传播到父节点
    fn backward(&self, parent: &Tensor) -> Vec<ArrayD<f32>> {
        // 处理广播情况
        let grad = parent.0.borrow().grad.clone();
        let grad_ref = grad.as_ref().expect("Gradient should not be None");
        let grad1 = grad.as_ref().expect("Gradient should not be None").clone();
        let grad2 = grad_ref.clone();

        vec![grad1, grad2]
    }
}

impl<'a, 'b> Add<&'a Tensor> for &'b Tensor {
    type Output = Tensor;

    fn add(self, other: &'a Tensor) -> Tensor {
        AddOp::new().forward(&[&self, &other])
    }
}
