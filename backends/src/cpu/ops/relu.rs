use core::{
    Layout,
    backends::{map::MapFunc, ops::Relu},
};
use num_traits::Zero;
use storage::cpu::{CpuDtype, CpuStorage};

use crate::cpu::CpuBackend;

#[derive(Debug)]
pub struct CpuRelu;

impl<U: CpuDtype + Zero + std::cmp::PartialOrd> MapFunc<CpuStorage<U>, CpuStorage<U>, U, U>
    for CpuRelu
{
    fn forward(&self, _layout: &Layout, storage: &CpuStorage<U>) -> CpuStorage<U> {
        let transformed_data: Vec<U> = storage
            .data
            .iter()
            .map(|x| if *x > U::zero() { x.clone() } else { U::zero() })
            .collect();

        CpuStorage {
            data: transformed_data,
        }
    }

    fn hint_string(&self) -> String {
        "Relu".to_string()
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
