use crate::backends::op_traits::ReduceFunc;
use crate::layout::Layout;
use crate::numeric::Zero;
use crate::op_traits::Sum;

use super::super::dtype::CpuDtype;
use super::super::storage::CpuStorage;

pub struct CpuSum;

impl<U: CpuDtype + Zero + std::ops::Add<Output = U>> ReduceFunc<U, U> for CpuSum {
    type InputStorage<A> = CpuStorage<U>;
    type OutputStorage<B> = CpuStorage<U>;

    fn call(&self, layout: &Layout, dim: i32, storage: &CpuStorage<U>) -> CpuStorage<U> {
        let actual_dim = if dim < 0 {
            (layout.shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        let mut output_shape = layout.shape.clone();
        output_shape.remove(actual_dim);

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
            for reduce_idx in 0..layout.shape[actual_dim] {
                let mut input_indices = output_indices.clone();
                input_indices.insert(actual_dim, reduce_idx);

                let input_flat_idx = layout.ravel_index(&input_indices);
                sum = sum + storage.data[input_flat_idx].clone();
            }

            output_data[output_idx] = sum;
        }

        CpuStorage { data: output_data }
    }
}

impl<T: CpuDtype> Sum for CpuStorage<T> {
    type Op = CpuSum;
    const OP: Self::Op = CpuSum;
}
