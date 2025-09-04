use core::{
    backends::{
        Backend,
        broadcast::BroadcastFunc,
        ops::{Multiply, Sum},
        reduce::ReduceFunc,
    },
    storage::Storage,
};

pub trait MatMul<S: Storage> {
    type TensorType;

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
        <B as Sum>::Sum: ReduceFunc<S, S, <S as Storage>::Inner, <S as Storage>::Inner> + 'static;
}
