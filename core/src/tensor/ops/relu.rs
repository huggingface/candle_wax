use crate::storage::Storage;
use crate::tensor::Tensor;
use crate::op_traits::Relu;

impl<S: Storage + Relu> Relu for Tensor<S>
{
    type Op = S::Op;
    const OP: Self::Op = S::OP;
}