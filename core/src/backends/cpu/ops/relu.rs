use crate::backends::cpu::backend::CpuBackend;
use crate::backends::op_traits::map::MapFunc;
use crate::backends::op_traits::map::relu::Relu;
use crate::layout::Layout;
use crate::numeric::Zero;
use crate::storage::cpu::{dtype::CpuDtype, storage::CpuStorage};

pub struct CpuRelu;

impl<U: CpuDtype + Zero + std::cmp::PartialOrd> MapFunc<CpuStorage<U>, CpuStorage<U>, U, U>
    for CpuRelu
{
    fn call(&self, _layout: &Layout, storage: &CpuStorage<U>) -> CpuStorage<U> {
        let transformed_data: Vec<U> = storage
            .data
            .iter()
            .map(|x| if *x > U::zero() { x.clone() } else { U::zero() })
            .collect();

        CpuStorage {
            data: transformed_data,
        }
    }
}

impl Default for CpuRelu {
    fn default() -> Self {
        CpuRelu
    }
}

impl Relu for CpuBackend {
    type Relu = CpuRelu;
}
