use std::sync::Arc;

pub trait Sum {
    type Sum: Default;

    fn as_arc() -> Arc<Self::Sum> {
        Arc::new(Self::Sum::default())
    }
}
