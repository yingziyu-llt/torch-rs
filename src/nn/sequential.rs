use crate::nn::Module;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
            // println!("output after layer {:?}: {:?}", layer, output);
        }
        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            if layer.parameters().is_empty() {
                continue;
            }
            params.extend(layer.parameters());
        }
        params
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}
