use crate::backends::Backend;
use crate::backends::{
    broadcast::{Broadcast, BroadcastFunc},
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

    pub fn broadcast<B, F, R, T>(
        self,
        other: &Tensor<R>,
        corresponding_dimensions: &[(i32, i32)],
        f: F,
    ) -> Tensor<T>
    where
        R: Storage,
        T: Storage,
        F: BroadcastFunc<S, R, T, S::Inner, R::Inner, T::Inner>,
        B: Backend + Broadcast<B, S, R, T, S::Inner, R::Inner, T::Inner, F>,
    {
        Tensor::new(
            self.layout.broadcast(
                &other.layout,
                &self
                    .layout
                    .signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
                        &other.layout,
                        corresponding_dimensions,
                    ),
            ),
            f.call(
                &self.layout,
                &self.storage,
                &other.layout,
                &other.storage,
                corresponding_dimensions,
            ),
        )
    }
}
