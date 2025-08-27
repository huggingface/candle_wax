use core::{
    Layout,
    backends::{
        Backend,
        map::{Map, MapFunc},
        reduce::{Reduce, ReduceFunc},
    },
    storage::Storage,
    tensor::{LazyTensor, Tensor},
};
use macros::BackendOps;

#[derive(BackendOps)]
pub struct CpuBackend {}

impl Backend for CpuBackend {
    fn eval<S: Storage>(tensor: LazyTensor<S>) -> Tensor<S> {
        match tensor {
            LazyTensor::Tensor(t) => t.as_ref().clone(),
            LazyTensor::Map { input, func } => {
                let new_tensor = Self::eval(*input);
                Tensor::new(
                    new_tensor.layout.clone(),
                    func.call(&new_tensor.layout, &new_tensor.storage),
                )
            }
            LazyTensor::Reduce { input, dim, func } => {
                let new_tensor = Self::eval(*input);
                Tensor::new(
                    new_tensor.layout.clone(),
                    func.call(&new_tensor.layout, &new_tensor.storage, dim),
                )
            }
        }
    }
}
