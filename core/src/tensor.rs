use std::marker::PhantomData;

use crate::backends::Backend;
use crate::backends::op_traits::{
    map::{Map, MapFunc},
    reduce::{Reduce, ReduceFunc},
};
use crate::layout::Layout;
use crate::storage::Storage;

pub struct Tensor<S: Storage, B: Backend> {
    pub layout: Layout,
    pub storage: S,
    backend: PhantomData<B>,
}

impl<S: Storage, B: Backend> Tensor<S, B> {
    pub fn new(layout: Layout, storage: S) -> Self {
        Self {
            layout,
            storage,
            backend: PhantomData,
        }
    }
}

impl<S: Storage, B: Backend> Tensor<S, B> {
    pub fn map<F, R>(self, f: F) -> Tensor<R, B>
    where
        R: Storage,
        F: MapFunc<S, R, S::Inner, R::Inner>,
        B: Map<B, S, R, S::Inner, R::Inner, F>,
    {
        B::map(&self, f)
    }
}

impl<S: Storage, B: Backend> Tensor<S, B> {
    pub fn reduce<F, R>(self, dim: i32, f: F) -> Tensor<R, B>
    where
        R: Storage,
        F: ReduceFunc<S, R, S::Inner, R::Inner>,
        B: Reduce<B, S, R, S::Inner, R::Inner, F>,
    {
        B::reduce(&self, dim, f)
    }
}
