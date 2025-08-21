use crate::storage::Storage;
use crate::tensor::Tensor;
use crate::op_traits::Sum;

impl<S: Storage + Sum> Sum for Tensor<S> {
    type Op = S::Op;
    const OP: Self::Op = S::OP;
}