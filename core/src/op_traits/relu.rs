pub trait Relu {
    type Op;
    const OP: Self::Op;

    fn op() -> Self::Op {
        Self::OP
    }
    fn relu(&self) -> Self::Op {
        Self::OP
    }
}
