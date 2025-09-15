use std::collections::HashSet;

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
}
