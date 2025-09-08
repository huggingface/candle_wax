use std::sync::Arc;

pub trait Divide {
    type Divide: Default;

    fn as_arc() -> Arc<Self::Divide> {
        Arc::new(Self::Divide::default())
    }
}
