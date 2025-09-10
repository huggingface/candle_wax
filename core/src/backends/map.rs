use std::fmt::Debug;

use crate::{backends::Backend, layout::Layout, storage::Storage};

pub type MapFuncSame<S> = dyn MapFunc<S, S, <S as Storage>::Inner, <S as Storage>::Inner>;

pub trait Map<B, S, T, U, V, F>
where
    B: Backend,
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: MapFunc<S, T, U, V>,
{
    fn map(layout: &Layout, storage: &S, f: F) -> T;
}

pub trait MapFunc<S, T, U, V>: Debug
where
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
{
    fn forward(&self, layout: &Layout, storage: &S) -> T;

    fn as_str(&self) -> String;
}
