use std::fmt::Debug;

use crate::{backends::Backend, layout::Layout, storage::Storage};
pub trait Broadcast<B, R, S, T, U, V, W, F>
where
    B: Backend,
    R: Storage<Inner = U>,
    S: Storage<Inner = V>,
    T: Storage<Inner = W>,
    F: BroadcastFunc<R, S, T, U, V, W>,
{
    fn broadcast(
        lhs_layout: &Layout,
        lhs_storage: &R,
        rhs_layout: &Layout,
        rhs_storage: &S,
        corresponding_dims: &[(i32, i32)],
        f: F,
    ) -> T;
}

pub trait BroadcastFunc<R, S, T, U, V, W>: Debug
where
    R: Storage<Inner = U>,
    S: Storage<Inner = V>,
    T: Storage<Inner = W>,
{
    fn forward(
        &self,
        lhs_layout: &Layout,
        lhs_storage: &R,
        rhs_layout: &Layout,
        rhs_storage: &S,
        corresponding_dims: &[(i32, i32)],
    ) -> T;

    fn as_str(&self) -> String;
}
