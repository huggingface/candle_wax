use std::sync::Arc;

use crate::Layout;
use crate::backends::{broadcast::BroadcastFuncSame, map::MapFuncSame, reduce::ReduceFuncSame};
use crate::storage::Storage;
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub enum LazyTensor<S: Storage> {
    Tensor(Arc<Tensor<S>>),
    Map {
        input: Box<LazyTensor<S>>,
        func: Arc<MapFuncSame<S>>,
        layout: Layout,
    },
    Reduce {
        input: Box<LazyTensor<S>>,
        dim: i32,
        func: Arc<ReduceFuncSame<S>>,
        layout: Layout,
    },
    Broadcast {
        lhs_input: Box<LazyTensor<S>>,
        rhs_input: Box<LazyTensor<S>>,
        corresponding_dimensions: Vec<(i32, i32)>,
        func: Arc<BroadcastFuncSame<S>>,
        layout: Layout,
    },
}

impl<S: Storage> From<Tensor<S>> for LazyTensor<S> {
    fn from(tensor: Tensor<S>) -> Self {
        LazyTensor::Tensor(Arc::new(tensor.clone()))
    }
}

impl<S: Storage> LazyTensor<S> {
    pub fn map(self, f: Arc<MapFuncSame<S>>) -> LazyTensor<S> {
        let new_layout = self.layout().clone();
        LazyTensor::Map {
            input: Box::new(self),
            func: f,
            layout: new_layout,
        }
    }

    pub fn reduce(self, dim: i32, f: Arc<ReduceFuncSame<S>>) -> LazyTensor<S> {
        let new_layout = self
            .layout()
            .reduce(self.layout().signed_dim_to_unsigned_dim(dim));
        LazyTensor::Reduce {
            input: Box::new(self),
            dim,
            func: f,
            layout: new_layout,
        }
    }

    pub fn broadcast(
        self,
        other: LazyTensor<S>,
        corresponding_dimensions: Vec<(i32, i32)>,
        f: Arc<BroadcastFuncSame<S>>,
    ) -> LazyTensor<S> {
        let new_layout = self.layout().broadcast(
            other.layout(),
            &self
                .layout()
                .signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
                    other.layout(),
                    &corresponding_dimensions,
                ),
        );
        LazyTensor::Broadcast {
            lhs_input: Box::new(self),
            rhs_input: Box::new(other),
            corresponding_dimensions,
            func: f,
            layout: new_layout,
        }
    }

    pub fn layout(&self) -> &Layout {
        match self {
            LazyTensor::Tensor(tensor) => &tensor.layout,
            LazyTensor::Map { layout, .. } => layout,
            LazyTensor::Reduce { layout, .. } => layout,
            LazyTensor::Broadcast { layout, .. } => layout,
        }
    }
}
