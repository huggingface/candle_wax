pub mod lazy_tensor;
pub mod ops;
pub mod tensor;

#[cfg(test)]
mod tests {
    use backends::cpu::CpuBackend;
    use core::{
        Layout,
        backends::LazyBackend,
        tensor::{LazyTensor, Tensor},
    };
    use storage::cpu::CpuStorage;

    use crate::ops::MatMul;

    #[test]
    fn test_composite_matmul_2d_lazy() {
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
        let result = CpuBackend::eval(tensor);

        assert_eq!(result.layout.shape, &[2, 2]);
        assert_eq!(result.storage.data, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
