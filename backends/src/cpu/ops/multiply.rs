use std::ops::Mul;

use crate::cpu::CpuBackend;
use core::{
    Layout,
    backends::{broadcast::BroadcastFunc, ops::Multiply},
    numeric::Zero,
};
use storage::cpu::{CpuDtype, CpuStorage};

pub struct CpuMultiply;

impl<U: CpuDtype + Zero + std::cmp::PartialOrd + Mul + Mul<Output = U>>
    BroadcastFunc<CpuStorage<U>, CpuStorage<U>, CpuStorage<U>, U, U, U> for CpuMultiply
{
    fn call(
        &self,
        lhs_layout: &Layout,
        lhs_storage: &CpuStorage<U>,
        rhs_layout: &Layout,
        rhs_storage: &CpuStorage<U>,
        corresponding_dims: &[(i32, i32)],
    ) -> CpuStorage<U> {
        let ucorresponding_dims = rhs_layout
            .signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
                lhs_layout,
                &corresponding_dims,
            );

        let output_layout = lhs_layout.broadcast(rhs_layout, &ucorresponding_dims);

        let output_size = output_layout.count_elements();
        let mut output_data = vec![U::zero(); output_size];

        for (output_idx, output_elem) in output_data.iter_mut().enumerate().take(output_size) {
            let output_indices = output_layout.unravel_index(output_idx);

            // This following converts output indices to the corresponding lhs and rhs indices
            // Since we know the layout should be the lhs_shape and then the non-corresponding
            // dimensions of the rhs_shape, we can just iterate through the output_indices and
            // pick out the correct indices for lhs and rhs.
            // So for example if lhs_shape = [2, 3, 4] and rhs_shape = [3, 4, 5]
            // and corresponding_dims = [(1, 0), (2, 1)]
            // then output_shape = [2, 3, 4, 5]
            // and output_indices = [i, j, k, l]
            // then lhs_indices = [i, j, k] and rhs_indices = [j, k, l]
            let mut lhs_indices = Vec::new();
            let mut rhs_indices = Vec::new();

            // First, handle the lhs dimensions (they come first in the output layout)
            for i in 0..lhs_layout.shape.len() {
                lhs_indices.push(output_indices[i]);
            }

            // Then, handle the rhs dimensions
            // We need to map from output dimensions to rhs dimensions using corresponding_dims
            let mut rhs_idx = 0;
            for i in 0..output_indices.len() {
                // Check if this output dimension corresponds to an lhs dimension
                let is_lhs_dim = i < lhs_layout.shape.len();

                if is_lhs_dim {
                    // Check if this lhs dimension has a corresponding rhs dimension
                    let lhs_dim = i;
                    if let Some(&(_, rhs_dim)) =
                        ucorresponding_dims.iter().find(|&&(l, _)| l == lhs_dim)
                    {
                        // This dimension is shared, use the same index
                        if rhs_indices.len() <= rhs_dim as usize {
                            rhs_indices.resize(rhs_dim as usize + 1, 0);
                        }
                        rhs_indices[rhs_dim as usize] = output_indices[i];
                    }
                } else {
                    // This is a non-corresponding rhs dimension
                    while rhs_indices.len() <= rhs_idx {
                        rhs_indices.push(0);
                    }
                    // Find the next non-corresponding rhs dimension
                    while ucorresponding_dims.iter().any(|&(_, r)| r == rhs_idx) {
                        rhs_idx += 1;
                    }
                    if rhs_idx < rhs_indices.len() {
                        rhs_indices[rhs_idx] = output_indices[i];
                    } else {
                        rhs_indices.push(output_indices[i]);
                    }
                    rhs_idx += 1;
                }
            }

            // Ensure rhs_indices has the correct length
            while rhs_indices.len() < rhs_layout.shape.len() {
                rhs_indices.push(0);
            }

            // Convert multidimensional indices to flat indices
            let lhs_flat_idx = lhs_layout.ravel_index(&lhs_indices);
            let rhs_flat_idx = rhs_layout.ravel_index(&rhs_indices);

            *output_elem =
                lhs_storage.data[lhs_flat_idx].clone() * rhs_storage.data[rhs_flat_idx].clone();
        }

        CpuStorage { data: output_data }
    }

    fn as_str(&self) -> String {
       "Multiply".to_string()
    }
}

impl Default for CpuMultiply {
    fn default() -> Self {
        CpuMultiply
    }
}

impl Multiply for CpuBackend {
    type Multiply = CpuMultiply;
}
