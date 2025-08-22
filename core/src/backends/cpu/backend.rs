use crate::backends::op_traits::{Map, MapFunc, Reduce, ReduceFunc};
use crate::storage::Storage;
use crate::tensor::Tensor;

pub struct CpuBackend {}

impl<S, T, U, V, F> Map<S, T, U, V, F> for CpuBackend
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: MapFunc<S, T, U, V>,
{

    fn map(tensor: &Tensor<S>, f: F) -> Tensor<T> {
        Tensor {
            layout: tensor.layout.clone(),
            storage: f.call(&tensor.layout, &tensor.storage),
        }
    }
}

impl <S, T, U, V, F> Reduce<S, T, U, V, F> for CpuBackend
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: ReduceFunc<S, T, U, V>,
{
    fn reduce(tensor: &Tensor<S>, dim: i32, f: F) -> Tensor<T> {
        let udim = tensor.layout.signed_dim_to_unsigned_dim(dim);
        Tensor {
            layout: tensor.layout.reduce(udim),
            storage: f.call(&tensor.layout, &tensor.storage, dim),
        }
    }
}