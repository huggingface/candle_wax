pub trait Sum {
    type Sum: Default;

    fn boxed() -> Box<Self::Sum> {
        Box::new(Self::Sum::default())
    }
}
