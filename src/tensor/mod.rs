use crate::autograd::op::Op;
use ndarray::ArrayD;
use std::fmt::{self, Debug};
use std::sync::{Arc, RwLock, Weak};

pub struct Tensor {
    pub data: Arc<ArrayD<f32>>,            //数据
    pub grad: RwLock<Option<ArrayD<f32>>>, // 梯度
    pub requires_grad: bool,
    pub creator: Option<Arc<dyn Op>>, // 创建该张量的操作
    pub parents: Vec<Weak<Tensor>>,   // 父节点
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 打印数据
        writeln!(f, "Tensor {{")?;
        writeln!(f, "  data: {:?}", self.data)?;

        // 打印梯度
        let grad = self.grad.read().unwrap();
        writeln!(f, "  grad: {:?}", grad)?;

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

impl Clone for Tensor {
    fn clone(&self) -> Self {
        // 克隆数据（共享所有权）
        let data = Arc::clone(&self.data);

        // 克隆梯度（需要处理锁）
        let grad = {
            let guard = self.grad.read().unwrap();
            RwLock::new(guard.clone())
        };

        Tensor {
            data,
            grad,
            requires_grad: self.requires_grad.clone(),
            creator: self.creator.clone(),
            parents: self.parents.clone(),
        }
    }
}

impl Tensor {
    pub fn new(data: ArrayD<f32>) -> Self {
        Tensor {
            data: Arc::new(data),
            grad: RwLock::new(None),
            requires_grad: false,
            creator: None,
            parents: Vec::new(),
        }
    }

    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    pub fn set_creator(&mut self, op: Arc<dyn Op>) {
        self.creator = Some(op);
    }

    pub fn add_parent(&mut self, parent: &Arc<Tensor>) {
        self.parents.push(Arc::downgrade(&Arc::clone(parent)));
    }
}
