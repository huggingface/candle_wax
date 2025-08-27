#[cfg(test)]
mod tests {
    use backends::cpu::CpuBackend;
    use core::{
        Layout,
        backends::{
            Backend,
            ops::{Relu, Sum},
        },
        tensor::{LazyTensor, Tensor},
    };
    use storage::cpu::CpuStorage;

    #[test]
    fn test_relu() {
        let tensor = Tensor::new(
            Layout::new(vec![5]),
            CpuStorage {
                data: vec![-1.0f32, 0.0, 1.0, -2.0, 3.0],
            },
        );

        let result = tensor.map::<CpuBackend, _, _>(<CpuBackend as Relu>::Relu::default());
        assert_eq!(result.storage.data, vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_sum_3d() {
        let tensor = Tensor::new(
            Layout::new(vec![2, 2, 2]),
            CpuStorage {
                data: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            },
        );

        let result = tensor.reduce::<CpuBackend, _, _>(2, <CpuBackend as Sum>::Sum::default());
        assert_eq!(result.storage.data, vec![3.0, 7.0, 11.0, 15.0]);
    }

    #[test]
    fn test_lazy() {
        let tensor = Tensor::new(
            Layout::new(vec![2, 2, 2]),
            CpuStorage {
                data: vec![1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0],
            },
        );

        let tensor = LazyTensor::from(tensor);
        let tensor = tensor.reduce(2, Box::new(<CpuBackend as Sum>::Sum::default()));
        let tensor = tensor.map(Box::new(<CpuBackend as Relu>::Relu::default()));
        let result = CpuBackend::eval(tensor);
        assert_eq!(result.storage.data, vec![3.0, 0.0, 11.0, 0.0]);
    }
}
