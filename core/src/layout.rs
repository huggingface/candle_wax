use std::{
    collections::HashSet,
    ops::{Bound, RangeBounds},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Layout {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl Layout {
    pub fn new(shape: Vec<usize>) -> Self {
        let strides = shape
            .iter()
            .rev()
            .scan(1, |acc, &dim| {
                let current = *acc;
                *acc *= dim;
                Some(current)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn count_dimensions(&self) -> usize {
        self.shape.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    pub fn count_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ravel_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "Indices length must match shape length"
        );
        indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&index, &stride)| index * stride)
            .sum::<usize>()
            + self.offset
    }

    pub fn unravel_index(&self, index: usize) -> Vec<usize> {
        let mut indices = vec![0; self.shape.len()];
        let mut remaining = index - self.offset;

        let mut stride_order: Vec<usize> = (0..self.strides.len()).collect();
        stride_order.sort_by_key(|&i| std::cmp::Reverse(self.strides[i]));

        for &dim in stride_order.iter() {
            indices[dim] = remaining / self.strides[dim];
            remaining %= self.strides[dim];
        }

        indices
    }

    pub fn reduce(&self, dim: usize) -> Self {
        assert!(dim < self.shape.len(), "Dimension out of bounds");
        let mut new_shape = self.shape.clone();
        new_shape.remove(dim);
        Layout::new(new_shape)
    }

    pub fn signed_dim_to_unsigned_dim(&self, dim: i32) -> usize {
        let udim = if dim < 0 {
            (self.shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };
        if self.shape.len() <= udim {
            panic!("Dimension out of bounds");
        }
        udim
    }

    pub fn broadcast(&self, other: &Layout, corresponding_dimensions: &[(usize, usize)]) -> Self {
        let mut this_duplicates = HashSet::new();
        let mut other_duplicates = HashSet::new();

        for &(self_dim, other_dim) in corresponding_dimensions.iter() {
            if this_duplicates.contains(&self_dim) || other_duplicates.contains(&other_dim) {
                panic!("Duplicate dimensions in corresponding_dimensions");
            }
            this_duplicates.insert(self_dim);
            other_duplicates.insert(other_dim);
        }

        // Check compatibility of corresponding dimensions
        for &(self_dim, other_dim) in corresponding_dimensions.iter() {
            if self.shape[self_dim] != other.shape[other_dim]
                && self.shape[self_dim] != 1
                && other.shape[other_dim] != 1
            {
                panic!("Incompatible dimensions for broadcasting");
            }
        }

        // The new layout should be the self layouts dimensions and the non-corresponding dimensions of other
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        for (i, &dim) in other.shape.iter().enumerate() {
            if !other_duplicates.contains(&i) {
                new_shape.push(dim);
                new_strides.push(other.strides[i]);
            }
        }

        Layout::new(new_shape)
    }

    pub fn signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
        &self,
        other: &Layout,
        corresponding_dimensions: &[(i32, i32)],
    ) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();

        for &(self_dim, other_dim) in corresponding_dimensions.iter() {
            let udim_self = self.signed_dim_to_unsigned_dim(self_dim);
            let udim_other = other.signed_dim_to_unsigned_dim(other_dim);
            pairs.push((udim_self, udim_other));
        }
        pairs
    }

    pub fn permute(&self, order: &[usize]) -> Self {
        assert_eq!(order.len(), self.shape.len(), "Shapes not the same length");
        // if the output vector must have every dim from 0-shape.len appear once
        let mut seen = vec![false; self.shape.len()];
        for &dim in order {
            assert!(!seen[dim], "Duplicate dimension in permutation");
            seen[dim] = true;
        }

        let mut new_shape = Vec::with_capacity(order.len());
        let mut new_strides = Vec::with_capacity(order.len());

        for &dim in order {
            new_shape.push(self.shape[dim]);
            new_strides.push(self.strides[dim]);
        }

        Layout {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        }
    }

    pub fn signed_dim_vec_to_unsigned_dim_vec(&self, dims: &[i32]) -> Vec<usize> {
        dims.iter()
            .map(|&dim| self.signed_dim_to_unsigned_dim(dim))
            .collect()
    }

    pub fn is_contiguous(&self) -> bool {
        // Check that strides are in decreasing order
        self.strides.windows(2).all(|w| w[0] >= w[1])
    }

    pub fn merge(&self, start_dim: usize, end_dim: usize) -> Self {
        assert!(
            self.is_contiguous(),
            "Layout must be contiguous to merge dimensions"
        );
        assert!(
            start_dim <= end_dim && end_dim < self.shape.len(),
            "Invalid dimension range: {} to {} for shape of length {}",
            start_dim,
            end_dim,
            self.shape.len()
        );

        let mut shape = Vec::new();
        shape.extend_from_slice(&self.shape[..start_dim]);
        shape.push(self.shape[start_dim..=end_dim].iter().product());
        if end_dim + 1 < self.shape.len() {
            shape.extend_from_slice(&self.shape[(end_dim + 1)..]);
        }

        let mut strides = Vec::new();
        strides.extend_from_slice(&self.strides[..start_dim]);
        strides.push(self.strides[end_dim]);
        if end_dim + 1 < self.shape.len() {
            strides.extend_from_slice(&self.strides[(end_dim + 1)..]);
        }

        Layout {
            shape,
            strides,
            offset: self.offset,
        }
    }

    pub fn signed_dim_range_to_unsigned_dim_range(
        &self,
        dims: impl RangeBounds<i32>,
    ) -> (usize, usize) {
        let start = match dims.start_bound() {
            Bound::Included(&dim) => self.signed_dim_to_unsigned_dim(dim),
            Bound::Excluded(&dim) => self.signed_dim_to_unsigned_dim(dim) + 1,
            Bound::Unbounded => 0,
        };
        let end = match dims.end_bound() {
            Bound::Included(&dim) => self.signed_dim_to_unsigned_dim(dim),
            Bound::Excluded(&dim) => self.signed_dim_to_unsigned_dim(dim) - 1,
            Bound::Unbounded => self.shape.len(),
        };
        assert!(
            start <= end && end <= self.shape.len(),
            "Invalid dimension range"
        );
        (start, end)
    }

    pub fn split(&self, dim: usize, sizes: &[usize]) -> Self {
        assert_eq!(
            self.shape[dim],
            sizes.iter().product(),
            "Sizes do not sum to dimension size"
        );

        let mut shape = Vec::new();
        shape.extend_from_slice(&self.shape[..dim]);
        shape.extend_from_slice(sizes);
        shape.extend_from_slice(&self.shape[(dim + 1)..]);

        let mut strides = Vec::new();
        strides.extend_from_slice(&self.strides[..dim]);
        strides.extend(
            sizes
                .iter()
                .rev()
                .scan(1, |acc, &dim| {
                    let current = *acc;
                    *acc *= dim;
                    Some(current)
                })
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|s| s * self.strides[dim]),
        );
        strides.extend_from_slice(&self.strides[(dim + 1)..]);

        Layout {
            shape,
            strides,
            offset: self.offset,
        }
    }

    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Self {
        assert!(
            start <= end && end < self.shape[dim],
            "Invalid slice bounds"
        );
        let mut shape = self.shape.clone();
        shape[dim] = end - start + 1;
        let strides = self.strides.clone();
        let offset = self.offset + start * self.strides[dim];
        Layout {
            shape,
            strides,
            offset,
        }
    }

    pub fn signed_index_to_unsigned_index(&self, dim: usize, index: i32) -> usize {
        let uindex = if index < 0 {
            (self.shape[dim] as i32 + index) as usize
        } else {
            index as usize
        };
        if self.shape[dim] <= uindex {
            panic!("Dimension out of bounds");
        }
        uindex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_dimensions() {
        assert_eq!(Layout::new(vec![2, 3, 4]).count_dimensions(), 3);
        assert_eq!(Layout::new(vec![5]).count_dimensions(), 1);
        assert_eq!(Layout::new(vec![]).count_dimensions(), 0);
    }

    #[test]
    fn test_is_scalar() {
        assert!(Layout::new(vec![]).is_scalar());
        assert!(!Layout::new(vec![1]).is_scalar());
        assert!(!Layout::new(vec![2, 3]).is_scalar());
    }

    #[test]
    fn test_count_elements() {
        assert_eq!(Layout::new(vec![2, 3, 4]).count_elements(), 24);
        assert_eq!(Layout::new(vec![5]).count_elements(), 5);
        assert_eq!(Layout::new(vec![]).count_elements(), 1);
        assert_eq!(Layout::new(vec![2, 0, 3]).count_elements(), 0);
    }

    #[test]
    fn test_ravel_index() {
        let layout = Layout::new(vec![2, 3, 4]);

        assert_eq!(layout.ravel_index(&[0, 0, 0]), 0);
        assert_eq!(layout.ravel_index(&[1, 2, 3]), 23);
        assert_eq!(layout.ravel_index(&[0, 1, 2]), 6);
        assert_eq!(layout.ravel_index(&[1, 0, 0]), 12);

        let layout_1d = Layout::new(vec![5]);
        assert_eq!(layout_1d.ravel_index(&[3]), 3);
    }

    #[test]
    #[should_panic(expected = "Indices length must match shape length")]
    fn test_ravel_index_wrong_length() {
        let layout = Layout::new(vec![2, 3, 4]);
        layout.ravel_index(&[0, 1]); // Wrong length
    }

    #[test]
    fn test_unravel_index() {
        let layout = Layout::new(vec![2, 3, 4]);

        assert_eq!(layout.unravel_index(0), vec![0, 0, 0]);
        assert_eq!(layout.unravel_index(23), vec![1, 2, 3]);
        assert_eq!(layout.unravel_index(6), vec![0, 1, 2]);
        assert_eq!(layout.unravel_index(12), vec![1, 0, 0]);

        let layout_1d = Layout::new(vec![5]);
        assert_eq!(layout_1d.unravel_index(3), vec![3]);
    }

    #[test]
    fn test_ravel_unravel_roundtrip() {
        let layout = Layout::new(vec![3, 4, 5]);

        for i in 0..layout.count_elements() {
            let indices = layout.unravel_index(i);
            let back_to_index = layout.ravel_index(&indices);
            assert_eq!(i, back_to_index);
        }
    }

    #[test]
    fn test_ravel_unravel_roundtrip_with_offset() {
        let mut layout = Layout::new(vec![3, 4, 5]);
        layout.offset = 10;

        for i in layout.offset..(layout.offset + layout.count_elements()) {
            let indices = layout.unravel_index(i);
            let back_to_index = layout.ravel_index(&indices);
            assert_eq!(
                i, back_to_index,
                "Failed roundtrip for index {} with offset {}",
                i, layout.offset
            );
        }
    }

    #[test]
    fn test_ravel_unravel_roundtrip_different_offsets() {
        let offsets = vec![0, 5, 17, 100];

        for offset in offsets {
            let mut layout = Layout::new(vec![2, 3, 4]);
            layout.offset = offset;

            for i in layout.offset..(layout.offset + layout.count_elements()) {
                let indices = layout.unravel_index(i);
                let back_to_index = layout.ravel_index(&indices);
                assert_eq!(
                    i, back_to_index,
                    "Failed roundtrip for index {} with offset {}",
                    i, offset
                );
            }
        }
    }

    #[test]
    fn test_ravel_unravel_roundtrip_custom_strides() {
        let test_cases = [
            Layout {
                shape: vec![3, 4],
                strides: vec![1, 3],
                offset: 0,
            },
            Layout {
                shape: vec![2, 3],
                strides: vec![10, 2],
                offset: 0,
            },
            Layout {
                shape: vec![2, 2, 2],
                strides: vec![8, 2, 1],
                offset: 5,
            },
        ];

        for (test_idx, layout) in test_cases.iter().enumerate() {
            let mut indices_combinations = Vec::new();
            generate_all_indices(&layout.shape, &mut vec![], &mut indices_combinations);

            for indices in indices_combinations {
                let flat_index = layout.ravel_index(&indices);
                let back_to_indices = layout.unravel_index(flat_index);
                assert_eq!(
                    indices, back_to_indices,
                    "Failed roundtrip for test case {} with indices {:?}, flat_index {}, layout {:?}",
                    test_idx, indices, flat_index, layout
                );
            }
        }
    }

    #[test]
    fn test_ravel_unravel_roundtrip_1d_with_custom_stride() {
        let layout = Layout {
            shape: vec![5],
            strides: vec![3],
            offset: 7,
        };

        for i in 0..layout.shape[0] {
            let indices = vec![i];
            let flat_index = layout.ravel_index(&indices);
            let expected_flat_index = layout.offset + i * layout.strides[0];
            assert_eq!(flat_index, expected_flat_index);

            let back_to_indices = layout.unravel_index(flat_index);
            assert_eq!(indices, back_to_indices);
        }
    }

    #[test]
    fn test_ravel_unravel_roundtrip_large_offset() {
        let mut layout = Layout::new(vec![2, 3]);
        layout.offset = 1000000;

        for i in layout.offset..(layout.offset + layout.count_elements()) {
            let indices = layout.unravel_index(i);
            let back_to_index = layout.ravel_index(&indices);
            assert_eq!(i, back_to_index);
        }
    }

    #[test]
    fn test_reduce() {
        let layout = Layout::new(vec![2, 3, 4]);

        let reduced_0 = layout.reduce(0);
        assert_eq!(reduced_0.shape, vec![3, 4]);
        assert_eq!(reduced_0.strides, vec![4, 1]);

        let reduced_1 = layout.reduce(1);
        assert_eq!(reduced_1.shape, vec![2, 4]);
        assert_eq!(reduced_1.strides, vec![4, 1]);

        let reduced_2 = layout.reduce(2);
        assert_eq!(reduced_2.shape, vec![2, 3]);
        assert_eq!(reduced_2.strides, vec![3, 1]);
    }

    #[test]
    #[should_panic(expected = "Dimension out of bounds")]
    fn test_reduce_out_of_bounds() {
        let layout = Layout::new(vec![2, 3]);
        layout.reduce(2);
    }

    #[test]
    fn test_signed_dim_to_unsigned_dim() {
        let layout = Layout::new(vec![2, 3, 4]);

        assert_eq!(layout.signed_dim_to_unsigned_dim(0), 0);
        assert_eq!(layout.signed_dim_to_unsigned_dim(1), 1);
        assert_eq!(layout.signed_dim_to_unsigned_dim(2), 2);

        assert_eq!(layout.signed_dim_to_unsigned_dim(-1), 2);
        assert_eq!(layout.signed_dim_to_unsigned_dim(-2), 1);
        assert_eq!(layout.signed_dim_to_unsigned_dim(-3), 0);
    }

    #[test]
    #[should_panic(expected = "Dimension out of bounds")]
    fn test_signed_dim_to_unsigned_dim_positive_out_of_bounds() {
        let layout = Layout::new(vec![2, 3]);
        layout.signed_dim_to_unsigned_dim(2);
    }

    #[test]
    #[should_panic(expected = "Dimension out of bounds")]
    fn test_signed_dim_to_unsigned_dim_negative_out_of_bounds() {
        let layout = Layout::new(vec![2, 3]);
        layout.signed_dim_to_unsigned_dim(-3);
    }

    #[test]
    fn test_broadcast_compatible() {
        let layout1 = Layout::new(vec![2, 3]);
        let layout2 = Layout::new(vec![3, 4]);

        let corresponding_dims = vec![(1, 0)];
        let broadcasted = layout1.broadcast(&layout2, &corresponding_dims);

        assert_eq!(broadcasted.shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_broadcast_with_size_1() {
        let layout1 = Layout::new(vec![2, 1]);
        let layout2 = Layout::new(vec![3, 4]);

        let corresponding_dims = vec![(1, 0)];
        let broadcasted = layout1.broadcast(&layout2, &corresponding_dims);

        assert_eq!(broadcasted.shape, vec![2, 1, 4]);
    }

    #[test]
    #[should_panic(expected = "Incompatible dimensions for broadcasting")]
    fn test_broadcast_incompatible() {
        let layout1 = Layout::new(vec![2, 3]);
        let layout2 = Layout::new(vec![4, 5]);

        let corresponding_dims = vec![(1, 0)];
        layout1.broadcast(&layout2, &corresponding_dims);
    }

    #[test]
    #[should_panic(expected = "Duplicate dimensions in corresponding_dimensions")]
    fn test_broadcast_duplicate_dimensions() {
        let layout1 = Layout::new(vec![2, 3]);
        let layout2 = Layout::new(vec![3, 4]);

        let corresponding_dims = vec![(0, 0), (0, 1)];
        layout1.broadcast(&layout2, &corresponding_dims);
    }

    #[test]
    fn test_broadcast_no_corresponding_dimensions() {
        let layout1 = Layout::new(vec![2, 3]);
        let layout2 = Layout::new(vec![4, 5]);

        let corresponding_dims = vec![];
        let broadcasted = layout1.broadcast(&layout2, &corresponding_dims);

        assert_eq!(broadcasted.shape, vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_signed_corresponding_dimensions_to_unsigned_corresponding_dimensions() {
        let layout1 = Layout::new(vec![2, 3, 4]);
        let layout2 = Layout::new(vec![4, 5]);

        let signed_dims = vec![(0, 0), (-1, -1)];
        let unsigned_dims = layout1
            .signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
                &layout2,
                &signed_dims,
            );

        assert_eq!(unsigned_dims, vec![(0, 0), (2, 1)]);
    }

    #[test]
    fn test_signed_corresponding_dimensions_mixed_signs() {
        let layout1 = Layout::new(vec![2, 3, 4]);
        let layout2 = Layout::new(vec![4, 5]);

        let signed_dims = vec![(1, -2), (-2, 0)];
        let unsigned_dims = layout1
            .signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
                &layout2,
                &signed_dims,
            );

        assert_eq!(unsigned_dims, vec![(1, 0), (1, 0)]);
    }

    #[test]
    fn test_layout_with_offset() {
        let mut layout = Layout::new(vec![2, 3]);
        layout.offset = 5;

        assert_eq!(layout.ravel_index(&[0, 0]), 5);
        assert_eq!(layout.ravel_index(&[1, 2]), 10);

        assert_eq!(layout.unravel_index(5), vec![0, 0]);
        assert_eq!(layout.unravel_index(10), vec![1, 2]);
    }

    #[test]
    fn test_layout_strides_calculation() {
        let test_cases = vec![
            (vec![2], vec![1]),
            (vec![3, 4], vec![4, 1]),
            (vec![2, 3, 4], vec![12, 4, 1]),
            (vec![5, 1, 3, 2], vec![6, 6, 2, 1]),
        ];

        for (shape, expected_strides) in test_cases {
            let layout = Layout::new(shape.clone());
            assert_eq!(
                layout.strides, expected_strides,
                "Failed for shape {:?}",
                shape
            );
        }
    }

    fn generate_all_indices(
        shape: &[usize],
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == shape.len() {
            result.push(current.clone());
            return;
        }

        let dim = current.len();
        for i in 0..shape[dim] {
            current.push(i);
            generate_all_indices(shape, current, result);
            current.pop();
        }
    }

    #[test]
    fn test_permute_identity() {
        let layout = Layout::new(vec![2, 3, 4]);
        let permuted = layout.permute(&[0, 1, 2]);

        assert_eq!(permuted.shape, vec![2, 3, 4]);
        assert_eq!(permuted.strides, vec![12, 4, 1]);
        assert_eq!(permuted.offset, 0);
    }

    #[test]
    fn test_permute_reverse() {
        let layout = Layout::new(vec![2, 3, 4]);
        let permuted = layout.permute(&[2, 1, 0]);

        assert_eq!(permuted.shape, vec![4, 3, 2]);
        assert_eq!(permuted.strides, vec![1, 4, 12]);
        assert_eq!(permuted.offset, 0);
    }

    #[test]
    fn test_permute_with_offset() {
        let mut layout = Layout::new(vec![2, 3, 4]);
        layout.offset = 10;
        let permuted = layout.permute(&[2, 0, 1]);

        assert_eq!(permuted.shape, vec![4, 2, 3]);
        assert_eq!(permuted.strides, vec![1, 12, 4]);
        assert_eq!(permuted.offset, 10);
    }

    #[test]
    #[should_panic(expected = "Shapes not the same length")]
    fn test_permute_wrong_length() {
        let layout = Layout::new(vec![2, 3, 4]);
        layout.permute(&[0, 1]);
    }

    #[test]
    #[should_panic(expected = "Duplicate dimension in permutation")]
    fn test_permute_duplicate() {
        let layout = Layout::new(vec![2, 3, 4]);
        layout.permute(&[0, 0, 1]);
    }

    #[test]
    fn test_signed_dim_vec_to_unsigned_dim_vec() {
        let layout = Layout::new(vec![2, 3, 4, 5]);

        let result = layout.signed_dim_vec_to_unsigned_dim_vec(&[0, -1, 2, -2]);
        assert_eq!(result, vec![0, 3, 2, 2]);

        let result2 = layout.signed_dim_vec_to_unsigned_dim_vec(&[-4, -3, -2, -1]);
        assert_eq!(result2, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_merge_adjacent_dimensions() {
        let layout = Layout::new(vec![2, 3, 4, 5]);
        let merged = layout.merge(1, 2);

        assert_eq!(merged.shape, vec![2, 12, 5]);
        assert_eq!(merged.strides, vec![60, 5, 1]);
        assert_eq!(merged.offset, 0);
    }

    #[test]
    fn test_merge_single_dimension() {
        let layout = Layout::new(vec![2, 3, 4]);
        let merged = layout.merge(1, 1);

        assert_eq!(merged.shape, vec![2, 3, 4]);
        assert_eq!(merged.strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_merge_all_dimensions() {
        let layout = Layout::new(vec![2, 3, 4]);
        let merged = layout.merge(0, 2);

        assert_eq!(merged.shape, vec![24]);
        assert_eq!(merged.strides, vec![1]);
    }

    #[test]
    fn test_merge_with_offset() {
        let mut layout = Layout::new(vec![2, 3, 4]);
        layout.offset = 5;
        let merged = layout.merge(0, 1);

        assert_eq!(merged.shape, vec![6, 4]);
        assert_eq!(merged.strides, vec![4, 1]);
        assert_eq!(merged.offset, 5);
    }

    #[test]
    #[should_panic(expected = "Invalid dimension range")]
    fn test_merge_invalid_range() {
        let layout = Layout::new(vec![2, 3, 4]);
        layout.merge(2, 1);
    }

    #[test]
    #[should_panic(expected = "Invalid dimension range: 0 to 3 for shape of length 3")]
    fn test_merge_out_of_bounds() {
        let layout = Layout::new(vec![2, 3, 4]);
        layout.merge(0, 3);
    }

    #[test]
    fn test_signed_dim_range_to_unsigned_dim_range() {
        let layout = Layout::new(vec![2, 3, 4, 5]);

        let (start, end) = layout.signed_dim_range_to_unsigned_dim_range(1..3);
        assert_eq!((start, end), (1, 2));

        let (start, end) = layout.signed_dim_range_to_unsigned_dim_range(-2..);
        assert_eq!((start, end), (2, 4));

        let (start, end) = layout.signed_dim_range_to_unsigned_dim_range(..=-1);
        assert_eq!((start, end), (0, 3));

        let (start, end) = layout.signed_dim_range_to_unsigned_dim_range(..);
        assert_eq!((start, end), (0, 4));
    }

    #[test]
    fn test_split_dimension() {
        let layout = Layout::new(vec![2, 6, 4]);
        let split_layout = layout.split(1, &[2, 3]);

        assert_eq!(split_layout.shape, vec![2, 2, 3, 4]);
        assert_eq!(split_layout.strides, vec![24, 12, 4, 1]);
        assert_eq!(split_layout.offset, 0);
    }

    #[test]
    fn test_split_into_three() {
        let layout = Layout::new(vec![12]);
        let split_layout = layout.split(0, &[3, 2, 2]);

        assert_eq!(split_layout.shape, vec![3, 2, 2]);
        assert_eq!(split_layout.strides, vec![4, 2, 1]);
    }

    #[test]
    fn test_split_with_offset() {
        let mut layout = Layout::new(vec![2, 8]);
        layout.offset = 10;
        let split_layout = layout.split(1, &[2, 4]);

        assert_eq!(split_layout.shape, vec![2, 2, 4]);
        assert_eq!(split_layout.strides, vec![8, 4, 1]);
        assert_eq!(split_layout.offset, 10);
    }

    #[test]
    #[should_panic(expected = "Sizes do not sum to dimension size")]
    fn test_split_size_mismatch() {
        let layout = Layout::new(vec![2, 6, 4]);
        layout.split(1, &[2, 2]);
    }

    #[test]
    fn test_split_merge_roundtrip() {
        let layout = Layout::new(vec![2, 6, 4]);
        let split_layout = layout.split(1, &[2, 3]);
        let merged_layout = split_layout.merge(1, 2);

        assert_eq!(merged_layout.shape, layout.shape);
        assert_eq!(merged_layout.strides, layout.strides);
        assert_eq!(merged_layout.offset, layout.offset);
    }

    #[test]
    fn test_broadcast_complex() {
        let layout1 = Layout::new(vec![2, 1, 3]);
        let layout2 = Layout::new(vec![4, 3, 1]);

        let corresponding_dims = vec![(2, 1)];
        let broadcasted = layout1.broadcast(&layout2, &corresponding_dims);

        assert_eq!(broadcasted.shape, vec![2, 1, 3, 4, 1]);
    }

    #[test]
    fn test_broadcast_multiple_corresponding() {
        let layout1 = Layout::new(vec![2, 3, 1]);
        let layout2 = Layout::new(vec![1, 4]);

        let corresponding_dims = vec![(0, 0), (2, 1)];
        let broadcasted = layout1.broadcast(&layout2, &corresponding_dims);

        assert_eq!(broadcasted.shape, vec![2, 3, 1]);
    }

    #[test]
    fn test_signed_operations_combined() {
        let layout = Layout::new(vec![2, 3, 4, 5]);

        let unsigned_dims = layout.signed_dim_vec_to_unsigned_dim_vec(&[-1, -2]);
        assert_eq!(unsigned_dims, vec![3, 2]);

        let (start, end) = layout.signed_dim_range_to_unsigned_dim_range(-3..-1);
        assert_eq!((start, end), (1, 2));

        let udim = layout.signed_dim_to_unsigned_dim(-4);
        assert_eq!(udim, 0);
    }

    #[test]
    fn test_edge_cases_empty_operations() {
        let layout = Layout::new(vec![1]);
        let merged = layout.merge(0, 0);
        assert_eq!(merged.shape, vec![1]);

        let split_layout = merged.split(0, &[1]);
        assert_eq!(split_layout.shape, vec![1]);
    }

    #[test]
    fn test_merge_split_permute_sequence() {
        let layout = Layout::new(vec![2, 3, 4, 2]);
        let merged = layout.merge(1, 2);
        let split_layout = merged.split(1, &[4, 3]);
        let permuted = split_layout.permute(&[0, 2, 1, 3]);

        assert_eq!(permuted.shape, vec![2, 3, 4, 2]);
        assert_eq!(permuted.count_elements(), 48);
    }

    #[test]
    fn test_split_merge_permute_roundtrip() {
        let original = Layout::new(vec![12, 2, 3]);
        let split_layout = original.split(0, &[3, 4]);
        let merged = split_layout.merge(0, 1);
        let permuted = merged.permute(&[0, 1, 2]);

        assert_eq!(permuted.shape, original.shape);
        assert_eq!(permuted.strides, original.strides);
    }

    #[test]
    fn test_split_multiple_dimensions_then_permute() {
        let layout = Layout::new(vec![24, 6]);
        let split1 = layout.split(0, &[4, 6]);
        let split2 = split1.split(2, &[2, 3]);
        let permuted = split2.permute(&[1, 3, 0, 2]);

        assert_eq!(permuted.shape, vec![6, 3, 4, 2]);
        assert_eq!(permuted.count_elements(), 144);
    }

    #[test]
    fn test_permute_then_split_multiple_dimensions() {
        let layout = Layout::new(vec![4, 6, 12]);
        let permuted = layout.permute(&[2, 0, 1]);
        assert_eq!(permuted.shape, vec![12, 4, 6]);
        assert_eq!(permuted.strides, vec![1, 72, 12]);

        let split1 = permuted.split(0, &[3, 4]);
        assert_eq!(split1.shape, vec![3, 4, 4, 6]);
        assert_eq!(split1.strides, vec![4, 1, 72, 12]);

        let split2 = split1.split(2, &[2, 2]);
        assert_eq!(split2.shape, vec![3, 4, 2, 2, 6]);
        assert_eq!(split2.strides, vec![4, 1, 144, 72, 12]);
    }

    #[test]
    fn test_merge_split_permute_with_offset() {
        let mut layout = Layout::new(vec![2, 4, 3, 2]);
        layout.offset = 100;

        let merged = layout.merge(1, 2);
        let split_layout = merged.split(1, &[6, 2]);
        let permuted = split_layout.permute(&[0, 2, 1, 3]);

        assert_eq!(permuted.offset, 100);
        assert_eq!(permuted.count_elements(), 48);

        for i in layout.offset..(layout.offset + layout.count_elements()) {
            let indices = layout.unravel_index(i);
            let flat = layout.ravel_index(&indices);
            assert_eq!(i, flat);
        }
    }

    #[test]
    fn test_alternating_split_merge_with_permute_retains_element_count() {
        let layout = Layout::new(vec![4, 6, 8]);
        let split1 = layout.split(1, &[2, 3]);
        let merged1 = split1.merge(0, 1);
        let permuted1 = merged1.permute(&[0, 2, 1]);
        let split2 = permuted1.split(0, &[2, 4]);

        assert_eq!(split2.count_elements(), layout.count_elements());
    }

    #[test]
    fn test_permute_preserves_split_merge_retains_element_count() {
        let layout = Layout::new(vec![6, 4, 9]);
        let split_layout = layout.split(2, &[3, 3]);

        for perm in &[
            vec![0, 1, 2, 3],
            vec![3, 2, 1, 0],
            vec![1, 0, 3, 2],
            vec![2, 3, 0, 1],
        ] {
            let permuted = split_layout.permute(perm);
            assert_eq!(permuted.count_elements(), layout.count_elements());
        }
    }

    #[test]
    fn test_identity_operations_chain() {
        let layout = Layout::new(vec![2, 3, 4]);
        let permuted = layout.permute(&[0, 1, 2]);
        let split_layout = permuted.split(1, &[3]);
        let merged = split_layout.merge(1, 1);

        assert_eq!(merged.shape, layout.shape);
        assert_eq!(merged.strides, layout.strides);
        assert_eq!(merged.offset, layout.offset);
    }

    #[test]
    fn test_split_first_dimension_multiple_ways() {
        let layout = Layout::new(vec![24, 5]);

        let split1 = layout.split(0, &[2, 12]);
        assert_eq!(split1.shape, vec![2, 12, 5]);
        assert_eq!(split1.strides, vec![60, 5, 1]);

        let split2 = layout.split(0, &[3, 8]);
        assert_eq!(split2.shape, vec![3, 8, 5]);
        assert_eq!(split2.strides, vec![40, 5, 1]);

        let split3 = layout.split(0, &[4, 6]);
        assert_eq!(split3.shape, vec![4, 6, 5]);
        assert_eq!(split3.strides, vec![30, 5, 1]);
    }

    #[test]
    fn test_merge_then_permute_multiple_orders() {
        let layout = Layout::new(vec![2, 3, 4, 5]);

        let merged1 = layout.merge(1, 2);
        let permuted1 = merged1.permute(&[0, 2, 1]);
        assert_eq!(permuted1.shape, vec![2, 5, 12]);

        let merged2 = layout.merge(0, 1);
        let permuted2 = merged2.permute(&[1, 0, 2]);
        assert_eq!(permuted2.shape, vec![4, 6, 5]);

        let merged3 = layout.merge(2, 3);
        let permuted3 = merged3.permute(&[2, 0, 1]);
        assert_eq!(permuted3.shape, vec![20, 2, 3]);
    }

    #[test]
    fn test_split_then_merge_different_ranges() {
        let layout = Layout::new(vec![2, 3, 4, 5, 6]);
        let split_layout = layout.split(2, &[2, 2]);
        assert_eq!(split_layout.shape, vec![2, 3, 2, 2, 5, 6]);

        let merged1 = split_layout.merge(1, 3);
        assert_eq!(merged1.shape, vec![2, 12, 5, 6]);

        let merged2 = split_layout.merge(0, 2);
        assert_eq!(merged2.shape, vec![12, 2, 5, 6]);

        let merged3 = split_layout.merge(4, 5);
        assert_eq!(merged3.shape, vec![2, 3, 2, 2, 30]);
    }

    #[test]
    fn test_split_multiple_dimensions_then_merge() {
        let layout = Layout::new(vec![60, 4]);
        let split1 = layout.split(0, &[5, 12]);
        let split2 = split1.split(1, &[3, 4]);

        assert_eq!(split2.shape, vec![5, 3, 4, 4]);
        assert_eq!(split2.strides, vec![48, 16, 4, 1]);

        let merged = split2.merge(1, 2);
        assert_eq!(merged.shape, vec![5, 12, 4]);
    }

    #[test]
    fn test_split_last_dimension_various_ways() {
        let layout = Layout::new(vec![3, 4, 12]);

        let split1 = layout.split(2, &[2, 6]);
        assert_eq!(split1.shape, vec![3, 4, 2, 6]);
        assert_eq!(split1.strides, vec![48, 12, 6, 1]);

        let split2 = layout.split(2, &[3, 4]);
        assert_eq!(split2.shape, vec![3, 4, 3, 4]);
        assert_eq!(split2.strides, vec![48, 12, 4, 1]);

        let split3 = layout.split(2, &[4, 3]);
        assert_eq!(split3.shape, vec![3, 4, 4, 3]);
        assert_eq!(split3.strides, vec![48, 12, 3, 1]);
    }

    #[test]
    fn test_merge_then_split_different_factorizations() {
        let layout = Layout::new(vec![2, 3, 4, 5, 6]);
        let merged_all = layout.merge(0, 4);

        assert_eq!(merged_all.shape, vec![720]);
        assert_eq!(merged_all.strides, vec![1]);

        let split1 = merged_all.split(0, &[8, 90]);
        assert_eq!(split1.shape, vec![8, 90]);

        let split2 = merged_all.split(0, &[24, 30]);
        assert_eq!(split2.shape, vec![24, 30]);

        let split3 = merged_all.split(0, &[9, 80]);
        assert_eq!(split3.shape, vec![9, 80]);
    }

    #[test]
    fn test_split_into_many_dimensions_then_permute() {
        let layout = Layout::new(vec![120]);
        let split_layout = layout.split(0, &[2, 3, 4, 5]);

        assert_eq!(split_layout.shape, vec![2, 3, 4, 5]);
        assert_eq!(split_layout.strides, vec![60, 20, 5, 1]);

        let permuted = split_layout.permute(&[3, 1, 0, 2]);
        assert_eq!(permuted.shape, vec![5, 3, 2, 4]);
        assert_eq!(permuted.strides, vec![1, 20, 60, 5]);
    }

    #[test]
    fn test_multiple_consecutive_splits_then_merge() {
        let layout = Layout::new(vec![144, 3]);
        let split1 = layout.split(0, &[12, 12]);
        let split2 = split1.split(0, &[3, 4]);
        let split3 = split2.split(2, &[2, 6]);

        assert_eq!(split3.shape, vec![3, 4, 2, 6, 3]);
        assert_eq!(split3.count_elements(), 432);

        let merged1 = split3.merge(0, 1);
        assert_eq!(merged1.shape, vec![12, 2, 6, 3]);

        let merged2 = split3.merge(2, 3);
        assert_eq!(merged2.shape, vec![3, 4, 12, 3]);
    }

    #[test]
    fn test_split_merge_same_dimension_roundtrip() {
        let layout = Layout::new(vec![2, 8, 3, 4]);
        let merged = layout.merge(1, 2);

        assert_eq!(merged.shape, vec![2, 24, 4]);

        let split_back = merged.split(1, &[8, 3]);
        assert_eq!(split_back.shape, vec![2, 8, 3, 4]);
        assert_eq!(split_back.strides, layout.strides);
    }

    #[test]
    fn test_split_different_dimensions_preserve_contiguity() {
        let layout = Layout::new(vec![5, 4, 6, 2]);

        let split1 = layout.split(0, &[1, 5]);
        assert_eq!(split1.shape, vec![1, 5, 4, 6, 2]);
        assert!(split1.is_contiguous());

        let split2 = layout.split(2, &[3, 2]);
        assert_eq!(split2.shape, vec![5, 4, 3, 2, 2]);
        assert!(split2.is_contiguous());
    }

    #[test]
    fn test_merge_overlapping_ranges() {
        let layout = Layout::new(vec![2, 24, 5]);
        let split_layout = layout.split(1, &[4, 6]);

        assert_eq!(split_layout.shape, vec![2, 4, 6, 5]);

        let merged1 = split_layout.merge(0, 2);
        assert_eq!(merged1.shape, vec![48, 5]);

        let merged2 = split_layout.merge(1, 3);
        assert_eq!(merged2.shape, vec![2, 120]);
    }

    #[test]
    fn test_split_merge_with_offset_preservation() {
        let mut layout = Layout::new(vec![3, 8, 4]);
        layout.offset = 25;

        let split_layout = layout.split(1, &[2, 4]);
        assert_eq!(split_layout.offset, 25);

        let merged = split_layout.merge(1, 2);
        assert_eq!(merged.offset, 25);
        assert_eq!(merged.shape, vec![3, 8, 4]);
    }

    #[test]
    fn test_permute_different_orders_preserve_elements() {
        let test_cases = vec![
            vec![6, 4, 5],
            vec![2, 9, 2, 3],
            vec![12, 8],
            vec![3, 3, 3, 3],
        ];

        for shape in test_cases {
            let layout = Layout::new(shape);
            let original_count = layout.count_elements();

            let permuted1 = layout.permute(&(0..layout.shape.len()).rev().collect::<Vec<_>>());
            assert_eq!(permuted1.count_elements(), original_count);

            let indices: Vec<usize> = (0..layout.shape.len()).collect();
            let permuted2 = layout.permute(&indices);
            assert_eq!(permuted2.count_elements(), original_count);
        }
    }

    #[test]
    fn test_split_single_element_dimensions() {
        let layout = Layout::new(vec![1, 12, 1, 6]);
        let split_layout = layout.split(1, &[3, 4]);

        assert_eq!(split_layout.shape, vec![1, 3, 4, 1, 6]);

        let merged = split_layout.merge(0, 2);
        assert_eq!(merged.shape, vec![12, 1, 6]);
    }

    #[test]
    fn test_split_dimensions_factorization() {
        let layout = Layout::new(vec![30, 8]);

        let split1 = layout.split(0, &[5, 6]);
        let split2 = split1.split(1, &[2, 3]);
        let split3 = split2.split(3, &[2, 4]);

        assert_eq!(split3.shape, vec![5, 2, 3, 2, 4]);
        assert_eq!(split3.count_elements(), 240);

        let merged = split3.merge(1, 2);
        assert_eq!(merged.shape, vec![5, 6, 2, 4]);
    }

    #[test]
    fn test_permute_identity_operations() {
        let layout = Layout::new(vec![2, 3, 4, 5]);
        let identity_perm = layout.permute(&[0, 1, 2, 3]);

        assert_eq!(identity_perm.shape, layout.shape);
        assert_eq!(identity_perm.strides, layout.strides);
        assert_eq!(identity_perm.offset, layout.offset);
    }

    #[test]
    fn test_merge_sequential_dimensions() {
        let layout = Layout::new(vec![2, 3, 4, 5, 6]);

        let merged1 = layout.merge(0, 2);
        assert_eq!(merged1.shape, vec![24, 5, 6]);

        let merged2 = layout.merge(2, 4);
        assert_eq!(merged2.shape, vec![2, 3, 120]);

        let merged3 = layout.merge(1, 3);
        assert_eq!(merged3.shape, vec![2, 60, 6]);
    }

    #[test]
    #[should_panic(expected = "Layout must be contiguous to merge dimensions")]
    fn test_merge_after_non_trivial_permute_fails() {
        let layout = Layout::new(vec![2, 3, 4]);
        let permuted = layout.permute(&[2, 0, 1]);
        permuted.merge(0, 1);
    }

    #[test]
    #[should_panic(expected = "Layout must be contiguous to merge dimensions")]
    fn test_merge_after_reverse_permute_fails() {
        let layout = Layout::new(vec![2, 3, 4, 5]);
        let permuted = layout.permute(&[3, 2, 1, 0]);
        permuted.merge(1, 2);
    }

    #[test]
    #[should_panic(expected = "Layout must be contiguous to merge dimensions")]
    fn test_merge_after_swap_permute_fails() {
        let layout = Layout::new(vec![4, 6]);
        let permuted = layout.permute(&[1, 0]);
        permuted.merge(0, 1);
    }

    #[test]
    fn test_merge_after_identity_permute_succeeds() {
        let layout = Layout::new(vec![2, 3, 4]);
        let permuted = layout.permute(&[0, 1, 2]);
        let merged = permuted.merge(1, 2);

        assert_eq!(merged.shape, vec![2, 12]);
    }

    #[test]
    fn test_complex_split_merge_chains() {
        let layout = Layout::new(vec![4, 6, 8, 3]);
        let split1 = layout.split(1, &[2, 3]);
        let merged1 = split1.merge(1, 2);
        let split2 = merged1.split(1, &[3, 2]);
        let merged2 = split2.merge(2, 3);

        assert_eq!(merged2.shape, vec![4, 3, 16, 3]);
        assert_eq!(merged2.count_elements(), layout.count_elements());
    }

    #[test]
    fn test_alternating_split_merge_operations() {
        let mut layout = Layout::new(vec![2, 12, 4]);
        layout.offset = 50;

        let original_elements = layout.count_elements();
        let original_offset = layout.offset;

        let split1 = layout.split(1, &[3, 4]);
        let merged1 = split1.merge(1, 2);
        let split2 = merged1.split(1, &[6, 2]);
        let merged2 = split2.merge(1, 2);

        assert_eq!(merged2.count_elements(), original_elements);
        assert_eq!(merged2.offset, original_offset);
        assert_eq!(merged2.shape, vec![2, 12, 4]);
    }

    #[test]
    fn test_split_then_permute_various_orders() {
        let layout = Layout::new(vec![24, 6]);
        let split_layout = layout.split(0, &[4, 6]);

        assert_eq!(split_layout.shape, vec![4, 6, 6]);

        let perm1 = split_layout.permute(&[1, 2, 0]);
        assert_eq!(perm1.shape, vec![6, 6, 4]);

        let perm2 = split_layout.permute(&[2, 0, 1]);
        assert_eq!(perm2.shape, vec![6, 4, 6]);

        let perm3 = split_layout.permute(&[0, 2, 1]);
        assert_eq!(perm3.shape, vec![4, 6, 6]);
    }

    #[test]
    fn test_slice_basic() {
        let layout = Layout::new(vec![5, 4, 3]);
        let sliced = layout.slice(0, 1, 4);

        assert_eq!(sliced.shape, vec![4, 4, 3]);
        assert_eq!(sliced.strides, vec![12, 3, 1]);
        assert_eq!(sliced.offset, 12); // 1 * stride[0]
    }

    #[test]
    fn test_slice_middle_dimension() {
        let layout = Layout::new(vec![3, 6, 2]);
        let sliced = layout.slice(1, 2, 5);

        assert_eq!(sliced.shape, vec![3, 4, 2]); // 6 - (5-2) + 1 = 4
        assert_eq!(sliced.strides, vec![12, 2, 1]);
        assert_eq!(sliced.offset, 4); // 2 * stride[1] = 2 * 2
    }

    #[test]
    fn test_slice_last_dimension() {
        let layout = Layout::new(vec![2, 3, 8]);
        let sliced = layout.slice(2, 3, 7);

        assert_eq!(sliced.shape, vec![2, 3, 5]);
        assert_eq!(sliced.strides, vec![24, 8, 1]);
        assert_eq!(sliced.offset, 3); // 3 * stride[2] = 3 * 1
    }

    #[test]
    fn test_slice_with_existing_offset() {
        let mut layout = Layout::new(vec![4, 6]);
        layout.offset = 15;
        let sliced = layout.slice(0, 1, 3);

        assert_eq!(sliced.shape, vec![3, 6]);
        assert_eq!(sliced.strides, vec![6, 1]);
        assert_eq!(sliced.offset, 21);
    }

    #[test]
    fn test_slice_single_element() {
        let layout = Layout::new(vec![5, 3]);
        let sliced = layout.slice(0, 2, 2);

        assert_eq!(sliced.shape, vec![1, 3]);
        assert_eq!(sliced.strides, vec![3, 1]);
        assert_eq!(sliced.offset, 6); // 2 * stride[0] = 2 * 3
    }

    #[test]
    #[should_panic(expected = "Invalid slice bounds")]
    fn test_slice_start_greater_than_end() {
        let layout = Layout::new(vec![5, 4]);
        layout.slice(0, 3, 2);
    }

    #[test]
    #[should_panic(expected = "Invalid slice bounds")]
    fn test_slice_end_out_of_bounds() {
        let layout = Layout::new(vec![5, 4]);
        layout.slice(0, 0, 5);
    }

    #[test]
    #[should_panic(expected = "Invalid slice bounds")]
    fn test_slice_start_out_of_bounds() {
        let layout = Layout::new(vec![5, 4]);
        layout.slice(1, 5, 4);
    }

    // Tests for signed_index_to_unsigned_index method
    #[test]
    fn test_signed_index_to_unsigned_index_positive() {
        let layout = Layout::new(vec![5, 3, 7]);

        assert_eq!(layout.signed_index_to_unsigned_index(0, 0), 0);
        assert_eq!(layout.signed_index_to_unsigned_index(0, 4), 4);
        assert_eq!(layout.signed_index_to_unsigned_index(1, 2), 2);
        assert_eq!(layout.signed_index_to_unsigned_index(2, 6), 6);
    }

    #[test]
    fn test_signed_index_to_unsigned_index_negative() {
        let layout = Layout::new(vec![5, 3, 7]);

        assert_eq!(layout.signed_index_to_unsigned_index(0, -1), 4); // 5 + (-1)
        assert_eq!(layout.signed_index_to_unsigned_index(0, -5), 0); // 5 + (-5)
        assert_eq!(layout.signed_index_to_unsigned_index(1, -1), 2); // 3 + (-1)
        assert_eq!(layout.signed_index_to_unsigned_index(1, -3), 0); // 3 + (-3)
        assert_eq!(layout.signed_index_to_unsigned_index(2, -7), 0); // 7 + (-7)
    }

    #[test]
    fn test_signed_index_to_unsigned_index_edge_cases() {
        let layout = Layout::new(vec![1, 10]);

        assert_eq!(layout.signed_index_to_unsigned_index(0, 0), 0);
        assert_eq!(layout.signed_index_to_unsigned_index(0, -1), 0);
        assert_eq!(layout.signed_index_to_unsigned_index(1, 9), 9);
        assert_eq!(layout.signed_index_to_unsigned_index(1, -10), 0);
    }

    #[test]
    #[should_panic(expected = "Dimension out of bounds")]
    fn test_signed_index_to_unsigned_index_positive_out_of_bounds() {
        let layout = Layout::new(vec![5, 3]);
        layout.signed_index_to_unsigned_index(0, 5);
    }

    #[test]
    #[should_panic(expected = "Dimension out of bounds")]
    fn test_signed_index_to_unsigned_index_negative_out_of_bounds() {
        let layout = Layout::new(vec![5, 3]);
        layout.signed_index_to_unsigned_index(1, -4);
    }

    // Tests for is_contiguous method edge cases
    #[test]
    fn test_is_contiguous_single_dimension() {
        let layout = Layout::new(vec![5]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_is_contiguous_scalar() {
        let layout = Layout::new(vec![]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_is_contiguous_custom_strides_non_contiguous() {
        let layout = Layout {
            shape: vec![2, 3],
            strides: vec![1, 2], // Non-standard order
            offset: 0,
        };
        assert!(!layout.is_contiguous());
    }

    #[test]
    fn test_is_contiguous_custom_strides_contiguous() {
        let layout = Layout {
            shape: vec![2, 3],
            strides: vec![6, 2], // Valid decreasing order
            offset: 0,
        };
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_is_contiguous_equal_strides() {
        let layout = Layout {
            shape: vec![2, 1, 3],
            strides: vec![3, 3, 1], // Equal strides are allowed
            offset: 0,
        };
        assert!(layout.is_contiguous());
    }

    // Integration tests combining multiple operations
    #[test]
    fn test_slice_then_permute() {
        let layout = Layout::new(vec![4, 5, 6]);
        let sliced = layout.slice(1, 1, 4);
        println!("Sliced layout: {:?}", sliced);
        let permuted = sliced.permute(&[2, 0, 1]);

        assert_eq!(permuted.shape, vec![6, 4, 4]);
        assert_eq!(permuted.offset, 6);
    }

    #[test]
    fn test_slice_then_split() {
        let layout = Layout::new(vec![6, 8]);
        let sliced = layout.slice(1, 2, 5);
        let split = sliced.split(1, &[2, 2]);

        assert_eq!(split.shape, vec![6, 2, 2]);
        assert_eq!(split.offset, 2); // Offset from slice
    }

    #[test]
    fn test_split_then_slice() {
        let layout = Layout::new(vec![4, 6]);
        let split = layout.split(1, &[2, 3]);
        let sliced = split.slice(1, 0, 1);

        assert_eq!(sliced.shape, vec![4, 2, 3]);
        assert_eq!(sliced.offset, 0);
    }

    #[test]
    fn test_multiple_slices() {
        let layout = Layout::new(vec![8, 6, 4]);
        let slice1 = layout.slice(0, 1, 7);
        let slice2 = slice1.slice(1, 1, 5);
        let slice3 = slice2.slice(2, 1, 3);

        assert_eq!(slice3.shape, vec![7, 5, 3]);
        assert_eq!(slice3.offset, 29);
    }

    #[test]
    fn test_slice_preserve_element_access() {
        let layout = Layout::new(vec![3, 4, 5]);
        let sliced = layout.slice(1, 1, 3);

        // Test that we can still access elements correctly
        let indices = vec![1, 0, 2];
        let original_flat = layout.ravel_index(&[1, 1, 2]); // Offset by slice start
        let sliced_flat = sliced.ravel_index(&indices);

        assert_eq!(original_flat, sliced_flat);
    }

    #[test]
    fn test_slice_unravel_consistency() {
        let layout = Layout::new(vec![4, 5, 3]);
        let sliced = layout.slice(1, 2, 4);

        // Test that ravel/unravel work correctly with sliced layout
        for i in 0..sliced.count_elements() {
            let indices = sliced.unravel_index(sliced.offset + i);
            let back_to_flat = sliced.ravel_index(&indices);
            assert_eq!(sliced.offset + i, back_to_flat);
        }
    }

    // Tests for error conditions and edge cases
    #[test]
    fn test_signed_index_boundary_conditions() {
        let layout = Layout::new(vec![1]);

        // Only valid indices for size 1 dimension are 0 and -1
        assert_eq!(layout.signed_index_to_unsigned_index(0, 0), 0);
        assert_eq!(layout.signed_index_to_unsigned_index(0, -1), 0);
    }

    #[test]
    fn test_complex_operations_with_slicing() {
        let mut layout = Layout::new(vec![6, 8, 4]);
        layout.offset = 10;

        let split = layout.split(1, &[2, 4]);
        let sliced = split.slice(2, 1, 3);
        let permuted = sliced.permute(&[0, 2, 1, 3]);

        assert_eq!(permuted.count_elements(), sliced.count_elements());
        assert!(permuted.offset >= layout.offset); // Offset should increase due to slice
    }

    #[test]
    fn test_slice_maintains_stride_relationships() {
        let layout = Layout::new(vec![5, 6, 7]);
        let sliced = layout.slice(1, 2, 5);

        assert_eq!(sliced.strides, layout.strides);

        assert_eq!(
            sliced.shape,
            vec![layout.shape[0], 5 - 2 + 1, layout.shape[2]]
        );
    }

    #[test]
    fn test_signed_index_with_various_dimension_sizes() {
        let sizes = vec![1, 2, 5, 10, 100];

        for size in sizes {
            let layout = Layout::new(vec![size]);

            // Test positive indices
            for i in 0..size {
                assert_eq!(layout.signed_index_to_unsigned_index(0, i as i32), i);
            }

            // Test negative indices
            for i in 1..=size {
                let signed_idx = -(i as i32);
                let expected = size - i;
                assert_eq!(
                    layout.signed_index_to_unsigned_index(0, signed_idx),
                    expected
                );
            }
        }
    }
}
