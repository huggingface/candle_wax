use crate::{layout::Layout, storage::Storage, tensor::Tensor};

pub mod sum;
pub use sum::Sum;

pub trait Reduce<S, T, U, V, F>
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: ReduceFunc<S, T, U, V>,
{
    fn reduce(tensor: &Tensor<S>, dim: i32, f: F) -> Tensor<T>;
}

pub trait ReduceFunc<S, T, U, V>
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
{
    fn call(&self, layout: &Layout, storage: &S, dim: i32) -> T;
}
