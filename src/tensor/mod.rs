use crate::ops::Op;
use ndarray::ArrayD;
use std::cell::RefCell;
use std::fmt::{self, Debug};
use std::rc::{Rc, Weak};

#[derive(Clone)]
pub struct TensorData {
    pub data: ArrayD<f32>,         //数据
    pub grad: Option<ArrayD<f32>>, // 梯度
    pub requires_grad: bool,
    pub creator: Option<Rc<dyn Op>>,             // 创建该张量的操作
    pub parents: Vec<Weak<RefCell<TensorData>>>, // 父节点
}

impl Debug for TensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 打印数据
        writeln!(f, "Tensor {{")?;
        writeln!(f, "  data: {:?}", self.data)?;

        // 打印梯度
        let grad_str = match &self.grad {
            Some(grad) => format!("{:?}", grad),
            None => "None".to_string(),
        };
        println!("  grad: {}", grad_str);

        // 打印 requires_grad
        writeln!(f, "  requires_grad: {}", self.requires_grad)?;

        // 打印 creator
        if let Some(creator) = &self.creator {
            writeln!(f, "  creator: {:?}", creator)?;
        } else {
            writeln!(f, "  creator: None")?;
        }

        // 打印 parents
        writeln!(f, "  parents: [")?;
        for (i, weak_ref) in self.parents.iter().enumerate() {
            match weak_ref.upgrade() {
                Some(strong_ref) => {
                    writeln!(f, "    Parent {}: {:?}", i, strong_ref)?;
                }
                None => {
                    writeln!(f, "    Parent {}: <dropped>", i)?;
                }
            }
        }
        writeln!(f, "  ]")?;

        write!(f, "}}")
    }
}

impl TensorData {
    pub fn new(data: ArrayD<f32>) -> Self {
        TensorData {
            grad: Some(ndarray::Array::zeros(data.raw_dim())),
            data,
            requires_grad: true,
            creator: None,
            parents: Vec::new(),
        }
    }

    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        if requires_grad == false {
            self.grad = None;
        } else {
            self.grad = Some(ndarray::Array::zeros(self.data.raw_dim()));
        }
        self
    }

    pub fn set_creator(&mut self, op: Rc<dyn Op>) {
        self.creator = Some(op);
    }

    pub fn add_parent(&mut self, parent: &Tensor) {
        self.parents.push(Rc::downgrade(&parent.0));
    }
}
#[derive(Clone, Debug)]
pub struct Tensor(pub Rc<RefCell<TensorData>>);

impl Tensor {
    pub fn new(data: ArrayD<f32>) -> Self {
        Tensor(Rc::new(RefCell::new(TensorData::new(data))))
    }
    pub fn requires_grad(self, requires_grad: bool) -> Self {
        self.0.borrow_mut().requires_grad = requires_grad;
        self
    }
}
