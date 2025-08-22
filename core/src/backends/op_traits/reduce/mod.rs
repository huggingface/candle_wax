use crate::{backends::Backend, layout::Layout, storage::Storage, tensor::Tensor};

pub mod sum;

pub trait Reduce<B, S, T, U, V, F>
where
    B: Backend,
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: ReduceFunc<S, T, U, V>,
{
    fn reduce(tensor: &Tensor<S, B>, dim: i32, f: F) -> Tensor<T, B>;
}

pub trait ReduceFunc<S, T, U, V>
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
{
    fn call(&self, layout: &Layout, storage: &S, dim: i32) -> T;
}
