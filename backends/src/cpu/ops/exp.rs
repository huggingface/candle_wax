use core::{
    Layout,
    backends::{map::MapFunc, ops::Exp},
};
use num_traits::real::Real;
use storage::cpu::{CpuDtype, CpuStorage};

use crate::cpu::CpuBackend;

#[derive(Debug)]
pub struct CpuExp;

impl<U: CpuDtype + Real + std::cmp::PartialOrd> MapFunc<CpuStorage<U>, CpuStorage<U>, U, U>
    for CpuExp
{
    fn forward(&self, _layout: &Layout, storage: &CpuStorage<U>) -> CpuStorage<U> {
        let transformed_data: Vec<U> = storage.data.iter().map(|x| x.exp()).collect();

        CpuStorage {
            data: transformed_data,
        }
    }

    fn as_str(&self) -> String {
        format!(
            "CpuExp({} -> {})",
            std::any::type_name::<CpuStorage<U>>(),
            std::any::type_name::<CpuStorage<U>>()
        )
    }
}

impl Default for CpuExp {
    fn default() -> Self {
        CpuExp
    }
}

impl Exp for CpuBackend {
    type Exp = CpuExp;
}
