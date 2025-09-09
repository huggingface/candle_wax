use core::Layout;
use num_traits::Zero;
use std::ops::{AddAssign, Mul};
use storage::cpu::{CpuDtype, CpuStorage};

#[derive(Debug)]
pub struct CpuMatmul;

pub trait Matmul<S> {
    fn forward(
        &self,
        lhs_layout: &Layout,
        lhs_storage: &S,
        rhs_layout: &Layout,
        rhs_storage: &S,
    ) -> S;
}

impl<
    U: CpuDtype + Zero + std::cmp::PartialOrd + Mul + Mul<Output = U> + AddAssign + std::fmt::Debug,
> Matmul<CpuStorage<U>> for CpuMatmul
{
    fn forward(
        &self,
        lhs_layout: &Layout,
        lhs_storage: &CpuStorage<U>,
        rhs_layout: &Layout,
        rhs_storage: &CpuStorage<U>,
    ) -> CpuStorage<U> {
        let lhs_udim_i = lhs_layout.signed_dim_to_unsigned_dim(-2);
        let lhs_udim_j = lhs_layout.signed_dim_to_unsigned_dim(-1);
        let rhs_udim_j = rhs_layout.signed_dim_to_unsigned_dim(-2);
        let rhs_udim_k = rhs_layout.signed_dim_to_unsigned_dim(-1);

        let reduced_size = lhs_layout.shape[lhs_udim_j];

        let mut out_shape = vec![];
        out_shape.extend_from_slice(&lhs_layout.shape[..lhs_udim_i]);
        out_shape.extend_from_slice(&rhs_layout.shape[..rhs_udim_j]);
        out_shape.push(lhs_layout.shape[lhs_udim_i]);
        out_shape.push(rhs_layout.shape[rhs_udim_k]);

        let output_layout = Layout::new(out_shape);

        let output_size = output_layout.count_elements();
        let mut output_data = vec![U::zero(); output_size];

        for (output_idx, output_elem) in output_data.iter_mut().enumerate().take(output_size) {
            let output_indices = output_layout.unravel_index(output_idx);

            let mut lhs_indices = Vec::new();
            let mut rhs_indices = Vec::new();

            let mut out_iter = output_indices.iter();

            for output_idx in (&mut out_iter).take(lhs_udim_i) {
                lhs_indices.push(*output_idx);
            }

            for output_idx in (&mut out_iter).take(rhs_udim_j) {
                rhs_indices.push(*output_idx);
            }

            let i_idx = (&mut out_iter).next().unwrap();
            let k_idx = (&mut out_iter).next().unwrap();

            for j_idx in 0..reduced_size {
                lhs_indices.push(*i_idx);
                lhs_indices.push(j_idx);

                rhs_indices.push(j_idx);
                rhs_indices.push(*k_idx);

                let lhs_flat_idx = lhs_layout.ravel_index(&lhs_indices);
                let rhs_flat_idx = rhs_layout.ravel_index(&rhs_indices);

                *output_elem +=
                    lhs_storage.data[lhs_flat_idx].clone() * rhs_storage.data[rhs_flat_idx].clone();

                lhs_indices.pop();
                lhs_indices.pop();
                rhs_indices.pop();
                rhs_indices.pop();
            }
        }

        CpuStorage { data: output_data }
    }
}

impl Default for CpuMatmul {
    fn default() -> Self {
        CpuMatmul
    }
}
