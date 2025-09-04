use core::{
    backends::{
        Backend,
        broadcast::BroadcastFunc,
        ops::{Multiply, Sum},
        reduce::ReduceFunc,
    },
    storage::Storage,
    tensor::LazyTensor,
};

use crate::ops::MatMul;

impl<S: Storage> MatMul<S> for LazyTensor<S> {
    type TensorType = LazyTensor<S>;

    fn matmul<B: Backend + Multiply + Sum>(self, other: Self::TensorType) -> Self::TensorType
    where
        <B as Multiply>::Multiply: BroadcastFunc<
                S,
                S,
                S,
                <S as Storage>::Inner,
                <S as Storage>::Inner,
                <S as Storage>::Inner,
            > + 'static,
        <B as Sum>::Sum: ReduceFunc<S, S, <S as Storage>::Inner, <S as Storage>::Inner> + 'static,
    {
        let tensor = self.broadcast(other, vec![(1, 0)], <B as Multiply>::as_arc());
        tensor.reduce(1, <B as Sum>::as_arc())
    }
}
