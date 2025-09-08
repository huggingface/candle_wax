use core::{
    backends::{broadcast::BroadcastFunc, map::MapFunc, reduce::ReduceFunc},
    storage::Storage,
    tensor::Tensor,
};
use egg::{CostFunction, EGraph, Extractor, Id, Rewrite, Runner};
use std::{collections::HashMap, sync::Arc};

use crate::{context::BackendContext, node_executor::EggNodeExecutor};

use super::{backend::CpuBackendError, language::CpuBackendLanguage, node_executor::CpuExecutor};

pub struct CpuBackendContext<S: Storage> {
    pub tensors: HashMap<usize, Arc<Tensor<S>>>,
    pub map_funcs: HashMap<String, Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>>,
    pub reduce_funcs: HashMap<String, Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>>,
    pub broadcast_funcs:
        HashMap<String, Arc<dyn BroadcastFunc<S, S, S, S::Inner, S::Inner, S::Inner>>>,
    pub corresponding_dims: HashMap<usize, Vec<(i32, i32)>>,

    pub egraph: EGraph<CpuBackendLanguage, ()>,
    pub eval_node_id: Option<Id>,
    pub rewrites: Vec<Rewrite<CpuBackendLanguage, ()>>,

    pub executor: CpuExecutor,
}

impl<S: Storage> BackendContext for CpuBackendContext<S> {
    type BackendStorage = S;
    type BackendError = CpuBackendError;
    type BackendLanguage = CpuBackendLanguage;

    fn set_eval_node_id(&mut self, id: Id) {
        self.eval_node_id = Some(id);
    }

    fn add_tensor(&mut self, tensor: Arc<Tensor<Self::BackendStorage>>) -> Id {
        let tensor_name = Arc::as_ptr(&tensor) as usize;
        self.tensors.insert(tensor_name, tensor);
        self.egraph.add(CpuBackendLanguage::Tensor(tensor_name))
    }

    fn add_map(
        &mut self,
        input: Id,
        func: Arc<
            dyn MapFunc<
                    Self::BackendStorage,
                    Self::BackendStorage,
                    <Self::BackendStorage as Storage>::Inner,
                    <Self::BackendStorage as Storage>::Inner,
                >,
        >,
    ) -> Id {
        let func_name = func.as_str();
        self.map_funcs.insert(func_name.clone(), func);
        let func_id = self.egraph.add(CpuBackendLanguage::MapFunc(func_name));
        self.egraph.add(CpuBackendLanguage::Map([input, func_id]))
    }

    fn add_reduce(
        &mut self,
        input: Id,
        dim: i32,
        func: Arc<
            dyn ReduceFunc<
                    Self::BackendStorage,
                    Self::BackendStorage,
                    <Self::BackendStorage as Storage>::Inner,
                    <Self::BackendStorage as Storage>::Inner,
                >,
        >,
    ) -> Id {
        let func_name = func.as_str();
        self.reduce_funcs.insert(func_name.clone(), func);
        let func_id = self.egraph.add(CpuBackendLanguage::ReduceFunc(func_name));
        let dim_id = self.egraph.add(CpuBackendLanguage::Dim(dim));
        self.egraph
            .add(CpuBackendLanguage::Reduce([input, dim_id, func_id]))
    }

    fn add_broadcast(
        &mut self,
        lhs_input: Id,
        rhs_input: Id,
        corresponding_dimensions: Vec<(i32, i32)>,
        func: Arc<
            dyn BroadcastFunc<
                    Self::BackendStorage,
                    Self::BackendStorage,
                    Self::BackendStorage,
                    <Self::BackendStorage as Storage>::Inner,
                    <Self::BackendStorage as Storage>::Inner,
                    <Self::BackendStorage as Storage>::Inner,
                >,
        >,
    ) -> Id {
        let func_name = func.as_str();
        self.broadcast_funcs.insert(func_name.clone(), func);
        let func_id = self
            .egraph
            .add(CpuBackendLanguage::BroadcastFunc(func_name));

        let lhs_dims = &corresponding_dimensions as *const _ as usize;
        self.corresponding_dims
            .insert(lhs_dims, corresponding_dimensions);

        let dims_id = self
            .egraph
            .add(CpuBackendLanguage::CorrespondingDims(lhs_dims));
        self.egraph.add(CpuBackendLanguage::Broadcast([
            lhs_input, rhs_input, dims_id, func_id,
        ]))
    }

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
    ) -> Result<Arc<Tensor<Self::BackendStorage>>, Self::BackendError> {
        let extractor = Extractor::new(&self.egraph, cost_func);
        let eval_id = self.eval_node_id.ok_or(CpuBackendError::NoEvaluationId)?;
        let (_cost, best_expr) = extractor.find_best(eval_id);
        let root = best_expr.last().ok_or(CpuBackendError::EmptyExpression)?;
        self.executor.execute_node(root, &best_expr, self)
    }
}

impl<S: Storage> Default for CpuBackendContext<S> {
    fn default() -> Self {
        Self {
            tensors: HashMap::new(),
            map_funcs: HashMap::default(),
            reduce_funcs: HashMap::default(),
            broadcast_funcs: HashMap::default(),
            corresponding_dims: HashMap::new(),
            egraph: EGraph::default(),
            eval_node_id: None,
            rewrites: Vec::new(),
            executor: CpuExecutor,
        }
    }
}
