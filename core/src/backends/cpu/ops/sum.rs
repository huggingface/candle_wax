use crate::backends::cpu::CpuBackend;
use crate::backends::op_traits::ReduceFunc;
use crate::backends::op_traits::Sum;
use crate::layout::Layout;
use crate::numeric::Zero;
use crate::storage::cpu::{CpuDtype, CpuStorage};

pub struct CpuSum;

impl<U: CpuDtype + Zero + std::ops::Add<Output = U>> ReduceFunc<CpuStorage<U>, CpuStorage<U>, U, U>
    for CpuSum
{
    fn call(&self, layout: &Layout, storage: &CpuStorage<U>, dim: i32) -> CpuStorage<U> {
        let udim = layout.signed_dim_to_unsigned_dim(dim);

        let mut output_shape = layout.shape.clone();
        output_shape.remove(udim);

        if output_shape.is_empty() {
            let total_sum = storage
                .data
                .iter()
                .fold(U::zero(), |acc, x| acc + x.clone());
            return CpuStorage {
                data: vec![total_sum],
            };
        }

        let output_layout = Layout::new(output_shape);
        let output_size = output_layout.shape.iter().product::<usize>();
        let mut output_data = vec![U::zero(); output_size];

        for output_idx in 0..output_size {
            let output_indices = output_layout.unravel_index(output_idx);

            let mut sum = U::zero();
            for reduce_idx in 0..layout.shape[udim] {
                let mut input_indices = output_indices.clone();
                input_indices.insert(udim, reduce_idx);

                let input_flat_idx = layout.ravel_index(&input_indices);
                sum = sum + storage.data[input_flat_idx].clone();
            }

            output_data[output_idx] = sum;
        }

        CpuStorage { data: output_data }
    }
}

impl Sum for CpuBackend {
    type Sum = CpuSum;
    const SUM: Self::Sum = CpuSum;
}
