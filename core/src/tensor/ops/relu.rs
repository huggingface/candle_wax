use crate::op_traits::Relu;
use crate::storage::Storage;
use crate::tensor::Tensor;

impl<S: Storage + Relu> Relu for Tensor<S> {
    type Op = S::Op;
    const OP: Self::Op = S::OP;
}
