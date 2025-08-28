#[cfg(test)]
mod tests {
    use backends::cpu::CpuBackend;
    use core::{
        Layout,
        backends::{
            LazyBackend,
            ops::{Multiply, Relu, Sum},
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
    fn test_matmul_2d_basic() {
        let tensor_a = Tensor::new(
            Layout::new(vec![2, 3]),
            CpuStorage {
                data: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        );

        let tensor_b = Tensor::new(
            Layout::new(vec![3, 2]),
            CpuStorage {
                data: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        );

        let multiplied = tensor_a.broadcast::<CpuBackend, _, _, _>(
            &tensor_b,
            &[(1, 0)],
            <CpuBackend as Multiply>::Multiply::default(),
        );
        let result = multiplied.reduce::<CpuBackend, _, _>(1, <CpuBackend as Sum>::Sum::default());

        assert_eq!(result.layout.shape, &[2, 2]);
        assert_eq!(result.storage.data, vec![22.0, 28.0, 49.0, 64.0]);
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
        let tensor = tensor.reduce(2, <CpuBackend as Sum>::as_arc());
        let tensor = tensor.map(<CpuBackend as Relu>::as_arc());
        let result = CpuBackend::eval(tensor);
        assert_eq!(result.storage.data, vec![3.0, 0.0, 11.0, 0.0]);
    }

    #[test]
    fn test_matmul_2d_lazy() {
        let tensor_a = Tensor::new(
            Layout::new(vec![2, 3]),
            CpuStorage {
                data: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        );

        let tensor_b = Tensor::new(
            Layout::new(vec![3, 2]),
            CpuStorage {
                data: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        );

        let tensor_a = LazyTensor::from(tensor_a);
        let tensor_b = LazyTensor::from(tensor_b);
        let tensor = tensor_a.broadcast(tensor_b, vec![(1,0)], <CpuBackend as Multiply>::as_arc());
        let tensor = tensor.reduce(1, <CpuBackend as Sum>::as_arc());
        let result = CpuBackend::eval(tensor);

        assert_eq!(result.layout.shape, &[2, 2]);
        assert_eq!(result.storage.data, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
