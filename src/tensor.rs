use crate::ops::Op;
use ndarray::{Array, ArrayD, Axis, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use std::cell::RefCell;
use std::fmt::{self, Debug};
use std::rc::Rc;

/// 张量数据结构，包含数据、梯度、依赖关系等。
#[derive(Clone)]
pub struct TensorData {
    /// 张量的实际数据
    pub data: ArrayD<f32>,
    /// 梯度
    pub grad: Option<ArrayD<f32>>,
    /// 是否需要计算梯度
    pub requires_grad: bool,
    /// 创建该张量的操作
    pub creator: Option<Rc<dyn Op>>,
    /// 父节点（依赖的张量）
    pub parents: Vec<Rc<RefCell<TensorData>>>,
}

// 实现Debug trait以便于调试输出
// 这里的实现类似于PyTorch的tensor输出格式
impl Debug for TensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor(")?;
        if self.data.len() <= 64 {
            write!(f, "{:?}", self.data)?;
        } else {
            write!(
                f,
                "[...tensor of size {}]",
                self.data
                    .shape()
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("×")
            )?;
        }

        // 显示梯度状态
        if self.requires_grad {
            write!(f, ", requires_grad=true")?;
        }

        // 显示梯度信息
        if let Some(ref grad) = self.grad {
            write!(f, ", grad={:?}", grad)?;
        } else {
            write!(f, ", grad=None")?;
        }

        // 显示创建者信息
        if let Some(ref creator) = self.creator {
            write!(f, ", creator={:?}", creator)?;
        } else {
            write!(f, ", creator=None")?;
        }

        write!(f, ")")
    }
}

impl TensorData {
    /// 创建新的张量数据
    pub fn new(data: ArrayD<f32>) -> Self {
        TensorData {
            grad: None,
            data,
            requires_grad: false,
            creator: None,
            parents: Vec::new(),
        }
    }

    /// 设置是否需要梯度
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        if requires_grad && self.grad.is_none() {
            self.grad = Some(ndarray::Array::zeros(self.data.raw_dim()));
        }
        self
    }

    /// 设置创建该张量的操作
    pub fn set_creator(&mut self, op: Rc<dyn Op>) {
        self.creator = Some(op);
    }

    /// 添加父节点
    pub fn add_parent(&mut self, parent: &Tensor) {
        self.parents.push(Rc::clone(&parent.0));
    }
}

/// 张量类型，自动微分的核心对象。
#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorData>>);

// 标准Debug实现
impl Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.borrow().fmt(f)
    }
}

impl Tensor {
    /// 用数据创建新张量
    pub fn new(data: ArrayD<f32>) -> Self {
        Tensor(Rc::new(RefCell::new(TensorData::new(data))))
    }

    /// 获取张量形状
    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().data.shape().to_vec()
    }

    /// 获取张量维度数
    pub fn dim(&self) -> usize {
        self.0.borrow().data.ndim()
    }

    /// 获取指定维度的长度
    pub fn size(&self, dim: usize) -> Option<usize> {
        if dim < self.dim() {
            Some(self.0.borrow().data.shape()[dim])
        } else {
            None
        }
    }

    /// 获取元素总数
    pub fn numel(&self) -> usize {
        self.0.borrow().data.len()
    }

    /// 沿第0维堆叠张量
    pub fn stack(tensors: &[Tensor]) -> Result<Tensor, &'static str> {
        if tensors.is_empty() {
            return Err("输入张量列表不能为空");
        }

        let first_shape = tensors[0].shape();
        for tensor in tensors.iter().skip(1) {
            if tensor.shape() != first_shape {
                return Err("所有张量的形状必须相同");
            }
        }

        // 先clone所有数据，避免生命周期问题
        let arrays: Vec<_> = tensors.iter().map(|t| t.0.borrow().data.clone()).collect();

        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();

        let stacked_data = ndarray::stack(Axis(0), &views).map_err(|_| "无法堆叠张量")?;
        Ok(Tensor::new(stacked_data))
    }

    /// 视图变换
    pub fn view(&self, shape: &[usize]) -> Result<Tensor, &'static str> {
        let borrowed = self.0.borrow();
        let total_elements = borrowed.data.len();
        let new_total = shape.iter().product();

        if total_elements != new_total {
            return Err("新形状的元素数量必须与原形状相同");
        }

        let reshaped_data = borrowed
            .data
            .clone()
            .into_shape(IxDyn(shape))
            .map_err(|_| "无法重新调整形状")?;

        Ok(Tensor::new(reshaped_data))
    }

    /// 重塑形状
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor, &'static str> {
        self.view(shape)
    }

    /// 设置是否需要梯度（链式调用）
    pub fn require_grad(&self, requires_grad: bool) -> Self {
        let mut borrowed = self.0.borrow_mut();
        borrowed.requires_grad = requires_grad;
        if requires_grad && borrowed.grad.is_none() {
            borrowed.grad = Some(ndarray::Array::zeros(borrowed.data.raw_dim()));
        }
        self.clone()
    }

    /// 是否为叶子节点
    pub fn is_leaf(&self) -> bool {
        self.0.borrow().creator.is_none()
    }

    /// 创建全0张量
    pub fn zeros(shape: &[usize]) -> Self {
        Tensor::new(ndarray::Array::zeros(IxDyn(shape))).require_grad(false)
    }

    /// 创建全1张量
    pub fn ones(shape: &[usize]) -> Self {
        Tensor::new(ndarray::Array::ones(IxDyn(shape))).require_grad(false)
    }

    /// 创建正态分布随机张量
    pub fn randn(shape: &[usize]) -> Self {
        Tensor::new(Array::random(IxDyn(shape), StandardNormal) / 100.0).require_grad(false)
    }

    /// 创建与自身形状相同的随机张量
    pub fn rand_like(&self) -> Self {
        Tensor::randn(&self.shape())
    }

    /// 创建与自身形状相同的全0张量
    pub fn zeros_like(&self) -> Self {
        Tensor::zeros(&self.shape())
    }

    /// 创建与自身形状相同的全1张量
    pub fn ones_like(&self) -> Self {
        Tensor::ones(&self.shape())
    }

    /// 获取数据副本
    pub fn data(&self) -> ArrayD<f32> {
        self.0.borrow().data.clone()
    }

    /// 获取标量值
    pub fn item(&self) -> Result<Self, &'static str> {
        let borrowed = self.0.borrow();
        if borrowed.data.ndim() != 0 {
            return Err("只能对标量张量调用item");
        }
        Ok(Tensor::new(borrowed.data.clone()))
    }

    /// 按索引取子张量
    pub fn index(&self, index: &[usize]) -> Result<Tensor, &'static str> {
        if index.len() != self.dim() {
            return Err("索引维度必须匹配张量维度");
        }

        let borrowed = self.0.borrow();
        let mut indices = Vec::new();

        for (i, &idx) in index.iter().enumerate() {
            if idx >= borrowed.data.shape()[i] {
                return Err("索引超出范围");
            }
            indices.push(idx);
        }

        // 简单实现，后续可以扩展
        let mut result = borrowed.data.clone();
        for (i, &idx) in indices.iter().enumerate().rev() {
            result = result.index_axis(Axis(i), idx).to_owned();
        }

        Ok(Tensor::new(result))
    }

    /// 返回不带梯度的新张量
    pub fn detach(&self) -> Self {
        // 创建一个新的TensorData，但共享原始数据（不进行深拷贝）
        let borrowed = self.0.borrow();

        // 创建一个新的TensorData对象，只共享数据
        let new_data = TensorData {
            data: borrowed.data.clone(), // 只克隆数据数组
            grad: None,                  // 不需要梯度
            requires_grad: false,        // 不需要计算梯度
            creator: None,               // 没有创建者
            parents: Vec::new(),         // 没有父节点
        };

        // 创建一个新的Tensor，包装新的TensorData
        Tensor(Rc::new(RefCell::new(new_data)))
    }

    /// 挤压指定或所有为1的维度
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Tensor, &'static str> {
        let borrowed = self.0.borrow();
        let mut new_shape: Vec<usize> = Vec::new();

        match dim {
            Some(d) => {
                if d >= borrowed.data.ndim() {
                    return Err("维度超出范围");
                }

                if borrowed.data.shape()[d] != 1 {
                    return Err("只能挤压大小为1的维度");
                }

                for (i, &size) in borrowed.data.shape().iter().enumerate() {
                    if i != d {
                        new_shape.push(size);
                    }
                }
            }
            None => {
                // 挤压所有大小为1的维度
                for &size in borrowed.data.shape() {
                    if size != 1 {
                        new_shape.push(size);
                    }
                }
            }
        }

        let reshaped = borrowed
            .data
            .clone()
            .into_shape(IxDyn(&new_shape))
            .map_err(|_| "重塑失败")?;

        Ok(Tensor::new(reshaped))
    }

    /// 在指定维度插入新轴
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor, &'static str> {
        let borrowed = self.0.borrow();
        let mut new_shape = borrowed.data.shape().to_vec();

        if dim > new_shape.len() {
            return Err("维度超出范围");
        }

        new_shape.insert(dim, 1);

        let reshaped = borrowed
            .data
            .clone()
            .into_shape(IxDyn(&new_shape))
            .map_err(|_| "重塑失败")?;

        Ok(Tensor::new(reshaped))
    }
}

impl From<ArrayD<f32>> for Tensor {
    fn from(data: ArrayD<f32>) -> Self {
        Tensor::new(data)
    }
}
impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        Tensor::new(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap())
    }
}
impl From<&[f32]> for Tensor {
    fn from(data: &[f32]) -> Self {
        Tensor::new(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap())
    }
}
