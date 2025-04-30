// src/main.rs
use ndarray::array;
use torch_rs::{ops::matmul::matmul, tensor::Tensor};

fn main() {
    let a = Tensor::new(array![[1.0], [2.0]].into_dyn()).requires_grad(true);
    let b = Tensor::new(array![[3.0], [4.0]].into_dyn()).requires_grad(true);
    let A = Tensor::new(array![[1.0, 0.0], [2.0, 3.0]].into_dyn()).requires_grad(true);

    let res = matmul(&A, &b);

    // println!("{:?}", c);
    // 反向传播
    res.backward();

    // println!("c:{:?}", c);
    println!("res:{:?}", res);
}
