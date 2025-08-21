use crate::backends::op_traits::ReduceFunc;

pub trait Reduce<U, V, F>
where
    F: ReduceFunc<U, V>,
{
    type OutStorage;

    fn reduce(&self, dim: i32, f: F) -> Self::OutStorage;
}
