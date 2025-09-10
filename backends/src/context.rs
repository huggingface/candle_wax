use core::{
    backends::{broadcast::BroadcastFuncSame, map::MapFuncSame, reduce::ReduceFuncSame},
    storage::Storage,
    tensor::Tensor,
};
use egg::{CostFunction, Id, Language, Rewrite};
use std::sync::Arc;

pub trait BackendContext: Default {
    type BackendStorage: Storage;
    type BackendError;
    type BackendLanguage: Language;

    fn set_eval_node_id(&mut self, id: Id);

    fn add_tensor(&mut self, tensor: Arc<Tensor<Self::BackendStorage>>) -> Id;

    fn add_map(&mut self, input: Id, func: Arc<MapFuncSame<Self::BackendStorage>>) -> Id;

    fn add_reduce(
        &mut self,
        input: Id,
        func: Arc<ReduceFuncSame<Self::BackendStorage>>,
        dim: i32,
    ) -> Id;

    fn add_broadcast(
        &mut self,
        lhs_input: Id,
        rhs_input: Id,
        func: Arc<BroadcastFuncSame<Self::BackendStorage>>,
        corresponding_dimensions: Vec<(i32, i32)>,
    ) -> Id;

    fn add_rewrites(&mut self, rewrites: &[Rewrite<Self::BackendLanguage, ()>]);

    fn optimize(&mut self);

    fn evaluate<C: CostFunction<Self::BackendLanguage>>(
        &mut self,
        cost_func: C,
    ) -> Result<Arc<Tensor<Self::BackendStorage>>, Self::BackendError>;
}
