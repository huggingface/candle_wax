use crate::storage::Storage;

use super::dtype::CpuDtype;

#[derive(Clone)]
pub struct CpuStorage<T: CpuDtype> {
    pub data: Vec<T>,
}

impl<T: CpuDtype> Storage for CpuStorage<T> {
    type Inner = T;
}
