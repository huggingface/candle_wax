use std::sync::Arc;

pub trait Relu {
    type Relu: Default;

    fn as_arc() -> Arc<Self::Relu> {
        Arc::new(Self::Relu::default())
    }
}
