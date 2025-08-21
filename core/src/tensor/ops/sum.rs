use crate::op_traits::Sum;
use crate::storage::Storage;
use crate::tensor::Tensor;

impl<S: Storage + Sum> Sum for Tensor<S> {
    type Op = S::Op;
    const OP: Self::Op = S::OP;
}
