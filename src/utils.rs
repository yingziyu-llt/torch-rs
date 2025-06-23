use crate::tensor::Tensor;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// 数据集 trait，类似 PyTorch 的 Dataset。
pub trait Dataset {
    /// 返回数据集长度
    fn len(&self) -> usize;
    /// 获取指定索引的数据和目标
    fn get(&self, idx: usize) -> (Tensor, Tensor);
}

/// 一个简单的张量数据集实现。
pub struct TensorDataset {
    /// 数据张量列表
    pub data: Vec<Tensor>,
    /// 目标张量列表
    pub targets: Vec<Tensor>,
}

impl TensorDataset {
    /// 创建新的张量数据集
    pub fn new(data: Vec<Tensor>, targets: Vec<Tensor>) -> Self {
        if data.len() != targets.len() {
            panic!("数据和目标的长度必须相同");
        }
        TensorDataset { data, targets }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        (self.data[idx].clone(), self.targets[idx].clone())
    }
}

/// DataLoader，支持 batch、shuffle、迭代。
pub struct DataLoader<'a, D: Dataset> {
    /// 数据集引用
    pub dataset: &'a D,
    /// batch 大小
    pub batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    idx: usize,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    /// 创建新的 DataLoader
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
            idx: 0,
        }
    }

    /// 重置迭代器，重新洗牌
    pub fn reset(&mut self) {
        self.idx = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }
}

impl<'a, D: Dataset> Iterator for DataLoader<'a, D> {
    type Item = (Tensor, Tensor);
    /// 获取下一个 batch
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.dataset.len() {
            return None;
        }
        let end = (self.idx + self.batch_size).min(self.dataset.len());
        let batch_idx = &self.indices[self.idx..end];
        let mut batch_data = Vec::with_capacity(batch_idx.len());
        let mut batch_targets = Vec::with_capacity(batch_idx.len());
        for &i in batch_idx {
            let (data, target) = self.dataset.get(i);
            batch_data.push(data);
            batch_targets.push(target);
        }
        let batch_data = Tensor::stack(&batch_data.as_slice())
            .unwrap()
            .require_grad(false);
        let batch_targets = Tensor::stack(&batch_targets.as_slice())
            .unwrap()
            .require_grad(false);
        self.idx = end;
        Some((batch_data, batch_targets))
    }
}
