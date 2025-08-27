use crate::backends::Backend;
use crate::backends::{
    map::{Map, MapFunc},
    reduce::{Reduce, ReduceFunc},
};
use crate::layout::Layout;
use crate::storage::Storage;

mod lazy_tensor;
pub use lazy_tensor::LazyTensor;

#[derive(Clone)]
pub struct Tensor<S: Storage> {
    pub layout: Layout,
    pub storage: S,
}

impl<S: Storage> Tensor<S> {
    pub fn new(layout: Layout, storage: S) -> Self {
        Self { layout, storage }
    }

    pub fn map<B, F, R>(self, f: F) -> Tensor<R>
    where
        R: Storage,
        F: MapFunc<S, R, S::Inner, R::Inner>,
        B: Backend + Map<B, S, R, S::Inner, R::Inner, F>,
    {
        Tensor::new(self.layout.clone(), f.call(&self.layout, &self.storage))
    }

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
