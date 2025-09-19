#[cfg(test)]
mod core_tests {
    use core::{Layout, tensor::Tensor};
    use storage::cpu::CpuStorage;

    #[test]
    fn test_permute() {
        let tensor = Tensor::new(
            Layout::new(vec![3, 2, 2]),
            CpuStorage {
                data: vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            },
        );

        let permuted = tensor.permute(vec![2, 0, 1]);
        assert_eq!(permuted.layout.shape, &[2, 3, 2]);
        assert_eq!(
            permuted.storage.data,
            vec![
                1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0
            ]
        );

        let contiguous = permuted.contiguous();
        assert_eq!(contiguous.layout.shape, &[2, 3, 2]);
        assert_eq!(
            contiguous.storage.data,
            vec![
                1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0
            ]
        );
    }

    #[test]
    fn test_merge_split() {
        let tensor = Tensor::new(
            Layout::new(vec![3, 2, 2]),
            CpuStorage {
                data: vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            },
        );

        let merge = tensor.merge(..=1);
        assert_eq!(merge.layout.shape, &[6, 2]);
        assert_eq!(
            merge.storage.data,
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]
        );

        let split = merge.split(0, vec![2, 3]);
        assert_eq!(split.layout.shape, &[2, 3, 2]);
        assert_eq!(
            split.storage.data,
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]
        );
    }
}

#[cfg(test)]
mod backend_ops_tests {
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
        let result = CpuBackend::eval(tensor).unwrap();
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
        let tensor = tensor_a.broadcast(tensor_b, vec![(1, 0)], <CpuBackend as Multiply>::as_arc());
        let tensor = tensor.reduce(1, <CpuBackend as Sum>::as_arc());
        let result = CpuBackend::eval(tensor).unwrap();

        assert_eq!(result.layout.shape, &[2, 2]);
        assert_eq!(result.storage.data, vec![22.0, 28.0, 49.0, 64.0]);
    }
}

#[cfg(test)]
mod composite_ops_tests {
    use backends::cpu::CpuBackend;
    use core::{
        Layout,
        backends::LazyBackend,
        tensor::{LazyTensor, Tensor},
    };
    use storage::cpu::CpuStorage;

    use composite_ops::{MatMul, Softmax};

    #[test]
    fn test_matmul() {
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
        let tensor = tensor_a.matmul::<CpuBackend>(tensor_b);
        let result = CpuBackend::eval(tensor).unwrap();

        assert_eq!(result.layout.shape, &[2, 2]);
        assert_eq!(result.storage.data, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_softmax() {
        let tensor_a = Tensor::new(
            Layout::new(vec![2, 3]),
            CpuStorage {
                data: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        );

        let tensor_a = LazyTensor::from(tensor_a);
        let tensor = tensor_a.softmax::<CpuBackend>();
        let result = CpuBackend::eval(tensor).unwrap();

        assert_eq!(result.layout.shape, &[2, 3]);
        assert_eq!(
            result.storage.data,
            vec![
                0.09003057,
                0.24472848,
                0.66524094,
                0.090030566,
                0.24472846,
                0.66524094
            ]
        );
    }
}
