use crate::cpu::CpuBackend;
use core::{
    Layout,
    backends::{map::MapFunc, ops::Relu},
    numeric::Zero,
};
use storage::cpu::{CpuDtype, CpuStorage};

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

    fn as_str(&self) -> String {
        format!(
            "CpuRelu({} -> {})",
            std::any::type_name::<CpuStorage<U>>(),
            std::any::type_name::<CpuStorage<U>>()
        )
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
