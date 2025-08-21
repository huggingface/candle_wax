pub mod dtype;
pub mod numeric;

pub mod layout;
pub mod storage;
pub mod tensor;

pub mod backends;

#[cfg(test)]
mod tests {
    use crate::backends::cpu::storage::CpuStorage;
    use crate::layout::Layout;
    use crate::tensor::{Map, Reduce, Tensor, Relu, Sum};

    #[test]
    fn test_relu() {
        let tensor = Tensor {
            layout: Layout::new(vec![5]),
            storage: CpuStorage {
                data: vec![-1.0f32, 0.0, 1.0, -2.0, 3.0],
            },
        };

        let result: Tensor<CpuStorage<f32>> = tensor.map(Relu::op(&tensor));
        assert_eq!(result.storage.data, vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_cross_type_conversion() {
        let tensor = Tensor {
            layout: Layout::new(vec![3]),
            storage: CpuStorage {
                data: vec![1i32, 2, 3],
            },
        };

        let result: Tensor<CpuStorage<f64>> = tensor.map(Relu::op(&tensor));
        assert_eq!(result.storage.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sum_3d() {
        let tensor = Tensor {
            layout: Layout::new(vec![2, 2, 2]),
            storage: CpuStorage {
                data: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            },
        };

        let result: Tensor<CpuStorage<f32>> = tensor.reduce(2, Sum::op(&tensor));
        assert_eq!(result.storage.data, vec![3.0, 7.0, 11.0, 15.0]);
    }
}
