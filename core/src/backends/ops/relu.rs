pub trait Relu {
    type Relu: Default;

    fn boxed() -> Box<Self::Relu> {
        Box::new(Self::Relu::default())
    }
}
