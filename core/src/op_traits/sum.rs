pub trait Sum {
    type Op;
    const OP: Self::Op;

    fn op() -> Self::Op {
        Self::OP
    }
    fn sum(&self) -> Self::Op {
        Self::OP
    }
}
