pub mod broadcast;
pub mod map;
pub mod reduce;

pub mod ops;

use crate::storage::Storage;
use crate::tensor::{LazyTensor, Tensor};

pub trait Backend {}

pub trait LazyBackend {
    fn eval<S: Storage>(tensor: LazyTensor<S>) -> Tensor<S>;
}
