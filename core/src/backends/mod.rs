pub mod map;
pub mod ops;
pub mod reduce;

use crate::storage::Storage;
use crate::tensor::{LazyTensor, Tensor};

pub trait Backend {
    fn eval<S: Storage>(tensor: LazyTensor<S>) -> Tensor<S>;
}
