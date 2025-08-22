use macros::BackendOps;

use crate::backends::Backend;
use crate::backends::op_traits::{Map, MapFunc, Reduce, ReduceFunc};
use crate::storage::Storage;
use crate::tensor::Tensor;

#[derive(BackendOps)]
pub struct CpuBackend {}

impl Backend for CpuBackend {}
