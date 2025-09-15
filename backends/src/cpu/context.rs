use core::{
    backends::{broadcast::BroadcastFuncSame, map::MapFuncSame, reduce::ReduceFuncSame},
    storage::Storage,
    tensor::Tensor,
};
use egg::{CostFunction, EGraph, Extractor, Id, Rewrite, Runner};
use std::{collections::HashMap, sync::Arc};

use crate::{
    context::{BackendContext, CoreContext},
    node_executor::EggNodeExecutor,
};

use super::{backend::CpuBackendError, language::CpuBackendLanguage, node_executor::CpuExecutor};

pub struct CpuBackendContext<S: Storage> {
    pub tensors: HashMap<usize, Arc<Tensor<S>>>,
    pub map_funcs: HashMap<usize, Arc<MapFuncSame<S>>>,
    pub reduce_funcs: HashMap<usize, Arc<ReduceFuncSame<S>>>,
    pub broadcast_funcs: HashMap<usize, Arc<BroadcastFuncSame<S>>>,

    pub egraph: EGraph<CpuBackendLanguage, ()>,
    pub eval_node_id: Option<Id>,
    pub rewrites: Vec<Rewrite<CpuBackendLanguage, ()>>,

    pub executor: CpuExecutor,
}

impl<S: Storage> BackendContext for CpuBackendContext<S> {
    type BackendStorage = S;
    type BackendError = CpuBackendError;
    type BackendLanguage = CpuBackendLanguage;

    fn add_rewrites(&mut self, other: &[Rewrite<CpuBackendLanguage, ()>]) {
        self.rewrites.extend_from_slice(other);
    }

    fn optimize(&mut self) {
        let runner = Runner::default()
            .with_egraph(std::mem::take(&mut self.egraph))
            .run(&self.rewrites);

        self.egraph = runner.egraph;
    }

    fn evaluate<C: CostFunction<Self::BackendLanguage>>(
        &mut self,
        cost_func: C,
        node_id: Id,
    ) -> Result<Arc<Tensor<Self::BackendStorage>>, Self::BackendError> {
        let extractor = Extractor::new(&self.egraph, cost_func);
        let (_cost, best_expr) = extractor.find_best(node_id);
        let root = best_expr.last().ok_or(CpuBackendError::EmptyExpression)?;
        self.executor.execute_node(root, &best_expr, self)
    }
}

impl<S: Storage> From<CoreContext<S>> for CpuBackendContext<S> {
    fn from(core: CoreContext<S>) -> Self {
        let egraph = core.map_egraph();
        let eval_node_id = core.last_added_id();
        Self {
            tensors: core.tensors,
            map_funcs: core.map_funcs,
            reduce_funcs: core.reduce_funcs,
            broadcast_funcs: core.broadcast_funcs,
            egraph,
            eval_node_id,
            rewrites: Vec::new(),
            executor: CpuExecutor,
        }
    }
}
