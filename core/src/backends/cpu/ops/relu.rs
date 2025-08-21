use crate::layout::Layout;
use crate::numeric::Zero;
use crate::backends::op_traits::MapFunc;
use crate::op_traits::Relu;

use super::super::dtype::CpuDtype;
use super::super::storage::CpuStorage;

pub struct CpuRelu;

impl<U: CpuDtype + Zero + std::cmp::PartialOrd> MapFunc<U, U>
    for CpuRelu
{
    type InputStorage<A> = CpuStorage<U>;
    type OutputStorage<B> = CpuStorage<U>;

    fn call(&self, _layout: &Layout, storage: &CpuStorage<U>) -> CpuStorage<U> {
        let transformed_data: Vec<U> = storage
            .data
            .iter()
            .map(|x| {
                if *x > U::zero() {
                    x.clone()
                } else {
                    U::zero()
                }
            })
            .collect();

        CpuStorage {
            data: transformed_data,
        }
    }
}

impl<T: CpuDtype> Relu for CpuStorage<T> {
    type Op = CpuRelu;
    const OP: Self::Op = CpuRelu;
}
