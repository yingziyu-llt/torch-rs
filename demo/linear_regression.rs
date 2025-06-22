use ndarray::array;
use torch_rs::nn::linear::Linear;
use torch_rs::nn::Module;
use torch_rs::optimizer::Optimizer;
use torch_rs::optimizer::SGD::SGD;
use torch_rs::tensor::Tensor;
use torch_rs::F;
// 线性回归demo
fn main() {
    let a = Tensor::new(
        array![
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 3.0],
            [5.0, 3.0],
            [5.0, 5.0],
            [10.0, 2.0]
        ]
        .into_dyn(),
    )
    .requires_grad(true);
    let b = Tensor::new(array![[2.0], [4.01], [5.99], [8.01], [10.005], [12.0]].into_dyn());
    let linear_layer = Linear::new(2, 1);
    let mut optimizer = SGD::new(linear_layer.parameters(), 0.01);
    let mut lr = 0.01 as f32;

    for i in 0..100 {
        let output = linear_layer.forward(&a);
        let loss = F::mse_loss(&output, &b);
        println!("Epoch {}: Loss = {:?}", i, loss);
        println!("Output: {:?}", output);

        // 反向传播
        loss.backward();

        // 更新参数
        optimizer.step();

        // 清零梯度
        optimizer.zero_grad();
        lr = lr * 0.99; // 学习率衰减
        optimizer.set_lr(lr);
    }
}
