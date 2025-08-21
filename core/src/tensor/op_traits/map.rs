use crate::backends::op_traits::MapFunc;

pub trait Map<U, V, F>
where
    F: MapFunc<U, V>,
{
    type OutStorage;

    fn map(&self, f: F) -> Self::OutStorage;
}