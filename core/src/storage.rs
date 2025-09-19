use crate::Layout;

pub trait Storage: Clone {
    type Inner: Clone;

    fn contiguous(&self, layout: &Layout) -> Self;
}
