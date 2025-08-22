use crate::{layout::Layout, storage::Storage, tensor::Tensor};

pub mod relu;
pub use relu::Relu;

pub trait Map<S, T, U, V, F>
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: MapFunc<S, T, U, V>,
{
    fn map(tensor: &Tensor<S>, f: F) -> Tensor<T>;
}

pub trait MapFunc<S, T, U, V>
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
{
    fn call(&self, layout: &Layout, storage: &S) -> T;
}
