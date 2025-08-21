use macros::TensorOps;

use crate::backends::op_traits::{MapFunc, ReduceFunc};
use crate::layout::Layout;
use crate::storage::Storage;

pub mod op_traits;
use op_traits::{Map, Reduce};

pub mod ops;

#[derive(TensorOps)]
pub struct Tensor<S: Storage> {
    pub layout: Layout,
    pub storage: S,
}

impl<S: Storage> Tensor<S> {
    // Common view operations that work on all tensor types
}
