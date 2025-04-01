// src/main.rs
use ndarray::array;
use std::sync::Arc;
use std::sync::Weak;
use torch_rs::autograd::add;
use torch_rs::tensor::Tensor;

fn print_graph(tensor: &Tensor, indent: usize) {
    println!("{:indent$}Tensor: {:?}", "", tensor.grad, indent = indent);
    for parent in &tensor.parents {
        if let Some(parent) = parent.upgrade() {
            print_graph(&parent, indent + 2);
        }
    }
}

fn inspect_parents(parents: &[Weak<Tensor>]) {
    for (i, weak_ref) in parents.iter().enumerate() {
        match weak_ref.upgrade() {
            Some(strong_ref) => {
                println!("Parent {}: {:?}", i, strong_ref);
                // 进一步检查父节点的内容
                println!("Parent {} data: {:?}", i, strong_ref.data);
            }
            None => {
                println!("Parent {} is invalid (already dropped)", i);
            }
        }
    }
}

fn main() {
    // 创建需要梯度的张量
    let a = Tensor::new(array![1.0, 2.0].into_dyn()).requires_grad(true);
    let b = Tensor::new(array![3.0, 4.0].into_dyn()).requires_grad(true);
    let arc_a = Arc::new(a);
    let arc_b = Arc::new(b);

    // 构建计算图：c = a + b
    let mut c = add(&arc_a, &arc_b);

    println!("{:?}", c.parents);
    inspect_parents(c.parents.as_slice());
    // 反向传播
    c.backward();

    println!("c:{:?}", c.data);
    println!("c的梯度: {:?}", c.grad);
    println!("a的梯度: {:?}", a.grad); // 应输出 [1.0, 1.0]
    println!("b的梯度: {:?}", b.grad); // 应输出 [1.0, 1.0]
}
