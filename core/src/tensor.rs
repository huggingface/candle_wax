use crate::backends::Backend;
use crate::backends::op_traits::{
    map::{Map, MapFunc},
    reduce::{Reduce, ReduceFunc},
};
use crate::layout::Layout;
use crate::storage::Storage;

pub struct Tensor<S: Storage> {
    pub layout: Layout,
    pub storage: S,
}

impl<S: Storage> Tensor<S> {
    pub fn new(layout: Layout, storage: S) -> Self {
        Self { layout, storage }
    }
}

impl<S: Storage> Tensor<S> {
    pub fn map<B, F, R>(self, f: F) -> Tensor<R>
    where
        R: Storage,
        F: MapFunc<S, R, S::Inner, R::Inner>,
        B: Backend + Map<B, S, R, S::Inner, R::Inner, F>,
    {
        Tensor::new(self.layout.clone(), f.call(&self.layout, &self.storage))
    }
}

impl<S: Storage> Tensor<S> {
    pub fn reduce<B, F, R>(self, dim: i32, f: F) -> Tensor<R>
    where
        R: Storage,
        F: ReduceFunc<S, R, S::Inner, R::Inner>,
        B: Backend + Reduce<B, S, R, S::Inner, R::Inner, F>,
    {
        Tensor::new(
            self.layout
                .reduce(self.layout.signed_dim_to_unsigned_dim(dim)),
            f.call(&self.layout, &self.storage, dim),
        )
    }
}
