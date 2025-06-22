// src/main.rs
use ndarray::array;
use torch_rs::tensor::Tensor;
use torch_rs::nn::linear::Linear;
use torch_rs::optimizer::SGD::SGD;
use torch_rs::optimizer::Optimizer;
use torch_rs::nn::Module;
use torch_rs::F;

fn main() {
    let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0],[5.0,6.0]].into_dyn());
    let b = Tensor::new(array![1.0,2.0,3.0].into_dyn());
    let linear_layer = Linear::new(2, 1);
    let mut optimizer = SGD::new(linear_layer.parameters(), 0.01);
    let output = linear_layer.forward(&a);
    let loss = F::mse_loss(&output, &b);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    println!("Output: {:?}", output);
    println!("Loss: {:?}", loss);
    println!("Weights: {:?}", linear_layer.w);
    println!("Bias: {:?}", linear_layer.b);
    let output = linear_layer.forward(&a);
    println!("Output after optimization: {:?}", output);
    println!("Final Loss: {:?}", F::mse_loss(&output, &b));
    println!("Final Weights: {:?}", linear_layer.w);
    println!("Final Bias: {:?}", linear_layer.b);
    println!("Training complete.");
}
