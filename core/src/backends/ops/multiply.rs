use std::sync::Arc;

pub trait Multiply {
    type Multiply: Default;

    fn as_arc() -> Arc<Self::Multiply> {
        Arc::new(Self::Multiply::default())
    }
}
