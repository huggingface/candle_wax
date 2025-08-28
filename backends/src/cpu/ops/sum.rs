use crate::cpu::CpuBackend;
use core::{
    Layout,
    backends::{ops::Sum, reduce::ReduceFunc},
    numeric::Zero,
};
use storage::cpu::{CpuDtype, CpuStorage};

pub struct CpuSum;

impl<U: CpuDtype + Zero + std::ops::Add<Output = U> + > ReduceFunc<CpuStorage<U>, CpuStorage<U>, U, U>
    for CpuSum
{
    fn call(&self, layout: &Layout, storage: &CpuStorage<U>, dim: i32) -> CpuStorage<U> {
        let udim = layout.signed_dim_to_unsigned_dim(dim);
        let output_layout = layout.reduce(udim);

        if output_layout.is_scalar() {
            let total_sum = storage
                .data
                .iter()
                .fold(U::zero(), |acc, x| acc + x.clone());
            return CpuStorage {
                data: vec![total_sum],
            };
        }

        let output_size = output_layout.count_elements();
        let mut output_data = vec![U::zero(); output_size];

        for (output_idx, output_elem) in output_data.iter_mut().enumerate().take(output_size) {
            let output_indices = output_layout.unravel_index(output_idx);

            let mut sum = U::zero();
            for reduce_idx in 0..layout.shape[udim] {
                let mut input_indices = output_indices.clone();
                input_indices.insert(udim, reduce_idx);

                let input_flat_idx = layout.ravel_index(&input_indices);
                sum = sum + storage.data[input_flat_idx].clone();
            }

            *output_elem = sum;
        }

        CpuStorage { data: output_data }
    }

    fn as_str(&self) -> String {
        "Sum".to_string()
    }
}

impl Default for CpuSum {
    fn default() -> Self {
        CpuSum
    }
}

impl Sum for CpuBackend {
    type Sum = CpuSum;
}
