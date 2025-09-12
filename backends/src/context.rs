use core::{storage::Storage, tensor::Tensor};
use egg::{CostFunction, Id, Language, Rewrite};
use std::sync::Arc;

pub trait BackendContext: Default {
    type BackendStorage: Storage;
    type BackendError;
    type BackendLanguage: Language;
    fn add_rewrites(&mut self, rewrites: &[Rewrite<Self::BackendLanguage, ()>]);

    fn optimize(&mut self);

    fn evaluate<C: CostFunction<Self::BackendLanguage>>(
        &mut self,
        cost_func: C,
        node_id: Id,
    ) -> Result<Arc<Tensor<Self::BackendStorage>>, Self::BackendError>;
}
