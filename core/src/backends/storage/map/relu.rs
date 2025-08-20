pub trait Relu {
    type Relu;
    fn op(&self) -> Self::Relu;
}