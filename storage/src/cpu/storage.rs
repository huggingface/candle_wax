use core::{Layout, storage::Storage};

use super::dtype::CpuDtype;

#[derive(Clone)]
pub struct CpuStorage<T: CpuDtype> {
    pub data: Vec<T>,
}

impl<T: CpuDtype> Storage for CpuStorage<T> {
    type Inner = T;

    #[expect(clippy::uninit_vec)]
    fn contiguous(&self, layout: &core::Layout) -> Self {
        let output_layout = Layout::new(layout.shape.clone());
        let mut output_data = Vec::with_capacity(layout.count_elements());
        unsafe {
            output_data.set_len(layout.count_elements());
        }

        for (output_idx, output_elem) in output_data.iter_mut().enumerate() {
            let output_idx_unraveled = output_layout.unravel_index(output_idx);
            let input_idx = layout.ravel_index(&output_idx_unraveled);
            *output_elem = self.data[input_idx].clone();
        }

        CpuStorage { data: output_data }
    }
}
