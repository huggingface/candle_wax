pub mod numeric;

mod layout;
pub use layout::Layout;

pub mod storage;

mod tensor;
pub use tensor::Tensor;

pub mod backends;

#[cfg(test)]
mod tests {
    use crate::backends::cpu::backend::CpuBackend;
    use crate::backends::op_traits::{map::relu::Relu, reduce::sum::Sum};
    use crate::layout::Layout;
    use crate::storage::cpu::storage::CpuStorage;
    use crate::tensor::Tensor;

    #[test]
    fn test_relu() {
        let tensor = Tensor::<_, CpuBackend>::new(
            Layout::new(vec![5]),
            CpuStorage {
                data: vec![-1.0f32, 0.0, 1.0, -2.0, 3.0],
            },
        );

        let result = tensor.map(CpuBackend::RELU);
        assert_eq!(result.storage.data, vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_sum_3d() {
        let tensor = Tensor::<_, CpuBackend>::new(
            Layout::new(vec![2, 2, 2]),
            CpuStorage {
                data: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            },
        );

        let result = tensor.reduce(2, CpuBackend::SUM);
        assert_eq!(result.storage.data, vec![3.0, 7.0, 11.0, 15.0]);
    }
}
