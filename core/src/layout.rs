#[derive(Debug, Clone)]
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
        let mut idx = index - self.offset;
        for (i, &stride) in self.strides.iter().enumerate() {
            indices[i] = idx / stride;
            idx %= stride;
        }
        indices
    }
}
