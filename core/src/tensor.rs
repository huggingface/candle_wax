use crate::backends::Backend;
use crate::backends::op_traits::{
    map::{Map, MapFunc},
    reduce::{Reduce, ReduceFunc},
};
use crate::layout::Layout;
use crate::storage::Storage;

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

pub enum LazyTensor<S: Storage> {
    Tensor(Tensor<S>),
    Map {
        input: Box<LazyTensor<S>>,
        func: Box<dyn MapFunc<S, S, S::Inner, S::Inner>>,
    },
    Reduce {
        input: Box<LazyTensor<S>>,
        dim: i32,
        func: Box<dyn ReduceFunc<S, S, S::Inner, S::Inner>>,
    },
}

impl<S: Storage> From<Tensor<S>> for LazyTensor<S> {
    fn from(tensor: Tensor<S>) -> Self {
        LazyTensor::Tensor(tensor)
    }
}

impl<S: Storage> LazyTensor<S> {
    pub fn map(self, f: Box<dyn MapFunc<S, S, S::Inner, S::Inner>>) -> LazyTensor<S> {
        LazyTensor::Map {
            input: Box::new(self),
            func: f,
        }
    }

    pub fn reduce(
        self,
        dim: i32,
        f: Box<dyn ReduceFunc<S, S, S::Inner, S::Inner>>,
    ) -> LazyTensor<S> {
        LazyTensor::Reduce {
            input: Box::new(self),
            dim,
            func: f,
        }
    }
}
