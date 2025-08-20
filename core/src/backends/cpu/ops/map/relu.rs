use crate::layout::Layout;
use crate::numeric::Zero;
use crate::backends::storage::{MapFunc, Relu};

use super::super::super::dtype::CpuDtype;
use super::super::super::storage::CpuStorage;

pub struct CpuRelu;

impl<U: CpuDtype + Zero + std::cmp::PartialOrd, V: CpuDtype + From<U> + Zero> MapFunc<U, V>
    for CpuRelu
{
    type InputStorage<A> = CpuStorage<U>;
    type OutputStorage<B> = CpuStorage<V>;

    fn call(&self, _layout: &Layout, storage: &CpuStorage<U>) -> CpuStorage<V> {
        let transformed_data: Vec<V> = storage
            .data
            .iter()
            .map(|x| {
                if *x > U::zero() {
                    V::from(x.clone())
                } else {
                    V::zero()
                }
            })
            .collect();

        CpuStorage {
            data: transformed_data,
        }
    }
}

impl<T: CpuDtype> Relu for CpuStorage<T> {
    type Relu = CpuRelu;

    fn op(&self) -> Self::Relu {
        CpuRelu
    }
}
