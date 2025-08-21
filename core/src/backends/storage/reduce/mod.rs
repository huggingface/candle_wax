use crate::layout::Layout;

pub mod sum;

pub trait Reduce<U, V, F>
where
    F: ReduceFunc<U, V>,
{
    type OutputStorage;

    fn reduce(&self, layout: &Layout, dim: i32, f: F) -> Self::OutputStorage;
}

pub trait ReduceFunc<U, V> {
    type InputStorage<A>;
    type OutputStorage<B>;

    fn call(
        &self,
        layout: &Layout,
        dim: i32,
        storage: &Self::InputStorage<U>,
    ) -> Self::OutputStorage<V>;
}
