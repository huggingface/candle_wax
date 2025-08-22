use crate::backends::op_traits::MapFunc;
use crate::layout::Layout;
use crate::numeric::Zero;
use crate::backends::op_traits::Relu;
use crate::storage::cpu::{
    CpuDtype,
    CpuStorage
};
use crate::backends::cpu::{
    CpuBackend
};

pub struct CpuRelu;

impl<U: CpuDtype + Zero + std::cmp::PartialOrd> MapFunc<CpuStorage<U>, CpuStorage<U>, U, U> for CpuRelu {

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

impl Relu for CpuBackend {
    type Relu = CpuRelu;
    const RELU: Self::Relu = CpuRelu;
}
