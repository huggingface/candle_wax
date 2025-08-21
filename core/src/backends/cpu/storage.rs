use macros::StorageOps;

use crate::backends::op_traits::{Map, MapFunc, Reduce, ReduceFunc};
use crate::layout::Layout;
use crate::storage::Storage;

use super::dtype::CpuDtype;

#[derive(StorageOps)]
pub struct CpuStorage<T: CpuDtype> {
    pub data: Vec<T>,
}

impl<T: CpuDtype> Storage for CpuStorage<T> {
    type Inner = T;
}
