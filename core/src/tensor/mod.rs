use macros::TensorOps;

use crate::layout::Layout;
use crate::storage::Storage;

pub mod op_traits;
pub mod ops;

#[derive(TensorOps)]
pub struct Tensor<S: Storage> {
    pub layout: Layout,
    pub storage: S,
}

impl<S: Storage> Tensor<S> {
    // Common view operations that work on all tensor types
}
