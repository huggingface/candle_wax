use std::ops::RangeBounds;

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

#[derive(Clone, Debug)]
pub struct Tensor<S: Storage> {
    pub layout: Layout,
    pub storage: S,
}

impl<S: Storage> Tensor<S> {
    pub fn new(layout: Layout, storage: S) -> Self {
        Self { layout, storage }
    }

    pub fn permute(&self, order: Vec<i32>) -> Tensor<S> {
        let new_layout = self
            .layout
            .permute(&self.layout.signed_dim_vec_to_unsigned_dim_vec(&order));
        Tensor::new(new_layout, self.storage.clone())
    }

    pub fn split(&self, dim: i32, sizes: Vec<usize>) -> Tensor<S> {
        let new_layout = self
            .layout
            .split(self.layout.signed_dim_to_unsigned_dim(dim), &sizes);
        Tensor::new(new_layout, self.storage.clone())
    }

    pub fn merge(&self, dim_range: impl RangeBounds<i32>) -> Tensor<S> {
        let (start_dim, end_dim) = self
            .layout
            .signed_dim_range_to_unsigned_dim_range(dim_range);
        let new_layout = self.layout.merge(start_dim, end_dim);
        Tensor::new(new_layout, self.storage.clone())
    }

    pub fn contiguous(&self) -> Tensor<S> {
        Tensor::new(
            Layout::new(self.layout.shape.clone()),
            self.storage.contiguous(&self.layout),
        )
    }

    pub fn map<B, F, R>(self, f: F) -> Tensor<R>
    where
        R: Storage,
        F: MapFunc<S, R, S::Inner, R::Inner>,
        B: Backend + Map<B, S, R, S::Inner, R::Inner, F>,
    {
        Tensor::new(self.layout.clone(), f.forward(&self.layout, &self.storage))
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
            f.forward(&self.layout, &self.storage, dim),
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
            f.forward(
                &self.layout,
                &self.storage,
                &other.layout,
                &other.storage,
                corresponding_dimensions,
            ),
        )
    }
}
