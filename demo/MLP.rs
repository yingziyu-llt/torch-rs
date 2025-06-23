use ndarray::array;
use torch_rs::functional;
use torch_rs::nn::Module;
use torch_rs::nn::linear::Linear;
use torch_rs::nn::sequential::Sequential;
use torch_rs::optimizer::Optimizer;
use torch_rs::optimizer::SGD::SGD;
use torch_rs::tensor::Tensor;
use torch_rs::utils::TensorDataset;

fn main() {
    let mut x = vec![];
    let mut y = vec![];
    let max_x = 2.0 * 50.0;
    let max_y = ((max_x * max_x) * 2.0) as f32;
    for i in 0..50 {
        let x_val = [i as f32, (i + 1) as f32];
        let y_val = (((i + (i + 1)) * (i + (i + 1))) as f32) * 2.0;
        x.push(
            Tensor::new(array![x_val[0] / max_x, x_val[1] / max_x].into_dyn()).require_grad(true),
        );
        y.push(Tensor::new(array![y_val / max_y].into_dyn()));
    }
    let dataset = TensorDataset::new(x, y);
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 64)),
        Box::new(torch_rs::nn::relu::ReLU::new()),
        Box::new(Linear::new(64, 1)),
    ]);
    let mut optimizer = SGD::new(model.parameters(), 0.01);
    for epoch in 0..100 {
        let output = model.forward(
            &Tensor::stack(dataset.data.as_slice())
                .unwrap()
                .require_grad(true),
        );
        let loss =
            functional::mse_loss(&output, &Tensor::stack(dataset.targets.as_slice()).unwrap());
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        let loss_total = loss.data().sum();
        println!("Epoch {}: Loss: {:?}", epoch, loss_total);
    }
}
