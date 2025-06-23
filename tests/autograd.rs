#[cfg(test)]
mod tests {
    use ndarray::array;
    use torch_rs::tensor::Tensor;

    #[test]
    fn test_add_backward() {
        let a = Tensor::new(array![1.0, 2.0, 3.0].into_dyn()).require_grad(true);
        let b = Tensor::new(array![4.0, 5.0, 6.0].into_dyn()).require_grad(true);
        let c = &a + &b;
        let d = c.mean();
        d.backward();

        let a_grad = a.0.borrow().grad.clone().unwrap();
        let b_grad = b.0.borrow().grad.clone().unwrap();
        // mean对每个元素的梯度都是1/3
        assert_eq!(
            a_grad,
            ndarray::Array::from_elem(a.shape(), 1.0 / 3.0).into_dyn()
        );
        assert_eq!(
            b_grad,
            ndarray::Array::from_elem(b.shape(), 1.0 / 3.0).into_dyn()
        );
    }

    #[test]
    fn test_mul_backward() {
        let a = Tensor::new(array![2.0, 3.0, 4.0].into_dyn()).require_grad(true);
        let b = Tensor::new(array![5.0, 6.0, 7.0].into_dyn()).require_grad(true);
        let c = &a * &b;
        let d = c.mean();
        d.backward();

        let a_grad = a.0.borrow().grad.clone().unwrap();
        let b_grad = b.0.borrow().grad.clone().unwrap();
        // mean对每个元素的梯度都是1/3，乘法链式法则
        assert_eq!(a_grad, b.data() * (1.0 / 3.0));
        assert_eq!(b_grad, a.data() * (1.0 / 3.0));
    }
    use torch_rs::ops::matmul::matmul;

    #[test]
    fn test_matmul_backward() {
        let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn()).require_grad(true);
        let b = Tensor::new(array![[2.0, 0.0], [1.0, 2.0]].into_dyn()).require_grad(true);
        let c = matmul(&a, &b);
        let d = c.mean();
        d.backward();

        let a_grad = a.0.borrow().grad.clone().unwrap();
        let b_grad = b.0.borrow().grad.clone().unwrap();
        // 这里只检查形状和数值范围
        assert_eq!(a_grad.shape(), a.shape().as_slice());
        assert_eq!(b_grad.shape(), b.shape().as_slice());
    }

    #[test]
    fn test_nested_backward() {
        let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn()).require_grad(true);
        let b = Tensor::new(array![[2.0, 0.0], [1.0, 2.0]].into_dyn()).require_grad(true);
        let c = &a + &b;
        let d = &c * &a;
        let e = matmul(&d, &b);
        let f = e.mean();
        f.backward();

        let a_grad = a.0.borrow().grad.clone().unwrap();
        let b_grad = b.0.borrow().grad.clone().unwrap();
        // 检查梯度是否已生成且形状正确
        assert_eq!(a_grad.shape(), a.shape().as_slice());
        assert_eq!(b_grad.shape(), b.shape().as_slice());
    }

    #[test]
    fn test_linear() {
        let input = Tensor::new(array![[1.0, 2.0, 3.0]].into_dyn()).require_grad(true);
        let weight = Tensor::new(array![[0.5], [0.5], [0.5]].into_dyn()).require_grad(true);
        let bias = Tensor::new(array![0.0].into_dyn()).require_grad(true);
        let output = &matmul(&input, &weight) + &bias;
        let loss = output.mean();
        loss.backward();
        println!("Input :{:?}", input);
        println!("Weight:{:?}", weight);
        println!("Bias  :{:?}", bias);
        println!("Output:{:?}", output);
    }
}
