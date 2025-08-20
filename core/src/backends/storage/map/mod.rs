use crate::layout::Layout;

pub mod relu;

pub trait Map<U, V, F>
where
    F: MapFunc<U, V>,
{
    type OutputStorage<T>;

    fn map(&self, layout: &Layout, f: F) -> Self::OutputStorage<V>;
}

pub trait MapFunc<U, V> {
    type InputStorage<A>;
    type OutputStorage<B>;

    fn call(&self, layout: &Layout, storage: &Self::InputStorage<U>) -> Self::OutputStorage<V>;
}