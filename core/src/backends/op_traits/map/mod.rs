use crate::{backends::Backend, layout::Layout, storage::Storage, tensor::Tensor};

pub mod relu;
pub use relu::Relu;

pub trait Map<B, S, T, U, V, F>
where
    B: Backend,
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: MapFunc<S, T, U, V>,
{
    fn map(tensor: &Tensor<S, B>, f: F) -> Tensor<T, B>;
}

pub trait MapFunc<S, T, U, V>
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
{
    fn call(&self, layout: &Layout, storage: &S) -> T;
}
