use macros::StorageOps;
use std::marker::PhantomData;

use crate::backends::op_traits::{Map, MapFunc, Reduce, ReduceFunc};
use crate::layout::Layout;
use crate::storage::Storage;

use super::dtype::MetalDtype;

#[derive(StorageOps)]
pub struct MetalStorage<T: MetalDtype>(pub PhantomData<T>);

impl<T: MetalDtype> Storage for MetalStorage<T> {
    type Inner = T;
}
