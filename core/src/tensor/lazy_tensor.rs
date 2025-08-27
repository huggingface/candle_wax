use std::sync::Arc;

use crate::backends::{map::MapFunc, reduce::ReduceFunc};
use crate::storage::Storage;
use crate::tensor::Tensor;

pub enum LazyTensor<S: Storage> {
    Tensor(Arc<Tensor<S>>),
    Map {
        input: Box<LazyTensor<S>>,
        func: Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>,
    },
    Reduce {
        input: Box<LazyTensor<S>>,
        dim: i32,
        func: Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>,
    },
}

impl<S: Storage> From<Tensor<S>> for LazyTensor<S> {
    fn from(tensor: Tensor<S>) -> Self {
        LazyTensor::Tensor(Arc::new(tensor.clone()))
    }
}

impl<S: Storage> LazyTensor<S> {
    pub fn map(self, f: Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>) -> LazyTensor<S> {
        LazyTensor::Map {
            input: Box::new(self),
            func: f,
        }
    }

    pub fn reduce(
        self,
        dim: i32,
        f: Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>,
    ) -> LazyTensor<S> {
        LazyTensor::Reduce {
            input: Box::new(self),
            dim,
            func: f,
        }
    }
}
