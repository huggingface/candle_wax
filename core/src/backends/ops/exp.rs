use std::sync::Arc;

pub trait Exp {
    type Exp: Default;

    fn as_arc() -> Arc<Self::Exp> {
        Arc::new(Self::Exp::default())
    }
}
