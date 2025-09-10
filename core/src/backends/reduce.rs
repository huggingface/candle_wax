use std::fmt::Debug;

use crate::{backends::Backend, layout::Layout, storage::Storage};

pub type ReduceFuncSame<S> = dyn ReduceFunc<S, S, <S as Storage>::Inner, <S as Storage>::Inner>;

pub trait Reduce<B, S, T, U, V, F>
where
    B: Backend,
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: ReduceFunc<S, T, U, V>,
{
    fn reduce(layout: &Layout, storage: &S, dim: i32, f: F) -> T;
}

pub trait ReduceFunc<S, T, U, V>: Debug
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
{
    fn forward(&self, layout: &Layout, storage: &S, dim: i32) -> T;

    fn as_str(&self) -> String;
}
