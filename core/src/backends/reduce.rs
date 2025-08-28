use crate::{backends::Backend, layout::Layout, storage::Storage};

pub trait Reduce<B, S, T, U, V, F>
where
    B: Backend,
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: ReduceFunc<S, T, U, V>,
{
    fn reduce(layout: &Layout, storage: &S, dim: i32, f: F) -> T;
}

pub trait ReduceFunc<S, T, U, V>
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
{
    fn call(&self, layout: &Layout, storage: &S, dim: i32) -> T;

    fn as_str(&self) -> String;
}
