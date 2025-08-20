pub trait Sum {
    type Sum;
    fn op(&self) -> Self::Sum;
}