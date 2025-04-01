use crate::Tensor;
use std::ops::Add;

impl Add for Tensor {
    type Output = Tensor;
    fn add(&self, b: &Arc<Tensor>) -> Tensor {
        let op = Arc::new(AddOp);
        let mut result = op.forward(&[self, b]);

        // 记录计算图关系
        result.add_parent(self);
        result.add_parent(b);
        result.set_creator(op.clone());

        result
    }
}
