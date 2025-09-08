use core::{
    backends::{
        Backend,
        broadcast::BroadcastFunc,
        map::MapFunc,
        ops::{Divide, Exp, Sum},
        reduce::ReduceFunc,
    },
    storage::Storage,
    tensor::LazyTensor,
};

pub trait Softmax<S: Storage> {
    type TensorType;

    fn softmax<B: Backend + Exp + Divide + Sum>(self) -> Self::TensorType
    where
        <B as Divide>::Divide: BroadcastFunc<
                S,
                S,
                S,
                <S as Storage>::Inner,
                <S as Storage>::Inner,
                <S as Storage>::Inner,
            > + 'static,
        <B as Exp>::Exp: MapFunc<S, S, <S as Storage>::Inner, <S as Storage>::Inner> + 'static,
        <B as Sum>::Sum: ReduceFunc<S, S, <S as Storage>::Inner, <S as Storage>::Inner> + 'static;
}

impl<S: Storage> Softmax<S> for LazyTensor<S> {
    type TensorType = LazyTensor<S>;

    fn softmax<B: Backend + Exp + Divide + Sum>(self) -> Self::TensorType
    where
        <B as Divide>::Divide: BroadcastFunc<
                S,
                S,
                S,
                <S as Storage>::Inner,
                <S as Storage>::Inner,
                <S as Storage>::Inner,
            > + 'static,
        <B as Exp>::Exp: MapFunc<S, S, <S as Storage>::Inner, <S as Storage>::Inner> + 'static,
        <B as Sum>::Sum: ReduceFunc<S, S, <S as Storage>::Inner, <S as Storage>::Inner> + 'static,
    {
        let exp_tensor = self.map(<B as Exp>::as_arc());
        let sum_tensor = exp_tensor.clone().reduce(-1, <B as Sum>::as_arc());
        exp_tensor.broadcast(sum_tensor, vec![(-2, -1)], <B as Divide>::as_arc())
    }
}
