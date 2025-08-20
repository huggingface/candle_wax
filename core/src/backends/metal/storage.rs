use macros::StorageOps;
use std::marker::PhantomData;

use crate::storage::Storage;

use super::dtype::MetalDtype;

#[derive(StorageOps)]
pub struct MetalStorage<T: MetalDtype> {
    data: PhantomData<T>,
}

impl<T: MetalDtype> Storage for MetalStorage<T> {
    type Inner = T;
}
