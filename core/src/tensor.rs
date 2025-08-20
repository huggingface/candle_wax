use crate::backends::storage::{MapFunc, ReduceFunc};
use crate::storage::Storage;
use crate::layout::Layout;

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
    type OutStorage<Q> : Storage<Inner = V>;

    fn map(&self, layout: &Layout, f: F) -> Tensor<Self::OutStorage<V>>;
}

impl<S, T, U, V, F> Map<U, V, F> for Tensor<S>
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: MapFunc<U, V, InputStorage<U> = S, OutputStorage<V> = T>,
{
    type OutStorage<Q> = T;

    fn map(&self, layout: &Layout, f: F) -> Tensor<Self::OutStorage<V>> {
        Tensor {
            layout: self.layout.clone(),
            storage: f.call(layout, &self.storage),
        }
    }
}

pub trait Reduce<U, V, F>
where
    F: ReduceFunc<U, V>,
{
    type OutStorage<Q> : Storage<Inner = V>;

    fn reduce(&self, layout: &Layout, dim: i32, f: F) -> Tensor<Self::OutStorage<V>>;
}

impl<S, T, U, V, F> Reduce<U, V, F> for Tensor<S>
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: ReduceFunc<U, V, InputStorage<U> = S, OutputStorage<V> = T>,
{
    type OutStorage<Q> = T;

    fn reduce(&self, layout: &Layout, dim: i32, f: F) -> Tensor<Self::OutStorage<V>> {
        Tensor {
            layout: self.layout.clone(),
            storage: f.call(layout, dim, &self.storage),
        }
    }
}