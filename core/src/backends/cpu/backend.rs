use crate::backends::Backend;
use crate::backends::op_traits::{Map, MapFunc, Reduce, ReduceFunc};
use crate::storage::Storage;
use crate::tensor::Tensor;

pub struct CpuBackend {}

impl<B, S, T, U, V, F> Map<B, S, T, U, V, F> for CpuBackend
where
    B: Backend,
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: MapFunc<S, T, U, V>,
{
    fn map(tensor: &Tensor<S, B>, f: F) -> Tensor<T, B> {
        let layout = tensor.layout.clone();
        let storage = f.call(&tensor.layout, &tensor.storage);
        Tensor::new(
            layout,
            storage
        )
    }
}

impl<B, S, T, U, V, F> Reduce<B, S, T, U, V, F> for CpuBackend
where
    B: Backend,
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: ReduceFunc<S, T, U, V>,
{
    fn reduce(tensor: &Tensor<S, B>, dim: i32, f: F) -> Tensor<T, B> {
        let layout = tensor.layout.clone();
        let storage = f.call(&tensor.layout, &tensor.storage, dim);
        Tensor::new(
            layout,
            storage
        )
    }
}

impl Backend for CpuBackend {}
