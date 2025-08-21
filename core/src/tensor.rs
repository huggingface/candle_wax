use macros::TensorOps;

use crate::backends::storage::{MapFunc, ReduceFunc, Relu as StorageRelu, Sum as StorageSum};
use crate::storage::Storage;
use crate::layout::Layout;

#[derive(TensorOps)]
pub struct Tensor<S: Storage> {
    pub layout: Layout,
    pub storage: S,
}

impl<S: Storage> Tensor<S> {    
    // Common view operations that work on all tensor types
}

pub trait Map<U, V, F>
where
    F: MapFunc<U, V>,
{
    type OutStorage;

    fn map(&self, f: F) -> Self::OutStorage;
}

pub trait Relu {
    type Relu;
    fn op(&self) -> Self::Relu;
}

impl<S: Storage + StorageRelu> Relu for Tensor<S>
{
    type Relu = S::Relu;

    fn op(&self) -> Self::Relu {
        S::op(&self.storage)
    }
}

pub trait Reduce<U, V, F>
where
    F: ReduceFunc<U, V>,
{
    type OutStorage;

    fn reduce(&self, dim: i32, f: F) -> Self::OutStorage;
}

pub trait Sum {
    type Sum;
    fn op(&self) -> Self::Sum;
}

impl<S: Storage + StorageSum> Sum for Tensor<S>
{
    type Sum = S::Sum;

    fn op(&self) -> Self::Sum {
        S::op(&self.storage)
    }
}