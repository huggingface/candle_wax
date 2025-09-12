use core::{
    backends::{broadcast::BroadcastFuncSame, map::MapFuncSame, reduce::ReduceFuncSame},
    storage::Storage,
    tensor::Tensor,
};
use egg::{CostFunction, EGraph, Extractor, Id, Rewrite, Runner};
use std::{collections::HashMap, sync::Arc};

use crate::{
    context::BackendContext,
    cpu::language::{FunctionLookup, TensorRef},
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

impl<S: Storage> CpuBackendContext<S> {
    pub fn set_eval_node_id(&mut self, id: Id) {
        self.eval_node_id = Some(id);
    }

    pub fn add_tensor(&mut self, tensor: Arc<Tensor<S>>) -> Id {
        let tensor_id = Arc::as_ptr(&tensor) as usize;
        let tensor_shape = tensor.layout.shape.clone();
        self.tensors.insert(tensor_id, tensor);
        self.egraph.add(CpuBackendLanguage::Tensor(TensorRef {
            id: tensor_id,
            shape: tensor_shape,
        }))
    }

    pub fn add_map(&mut self, input: Id, func: Arc<MapFuncSame<S>>) -> Id {
        let func_id = Arc::as_ptr(&func) as *const () as usize;
        let func_name = func.as_str();
        self.map_funcs.insert(func_id, func);
        let graph_func_id = self.egraph.add(CpuBackendLanguage::MapFunc(FunctionLookup {
            id: func_id,
            func_type: func_name,
        }));
        self.egraph
            .add(CpuBackendLanguage::Map([input, graph_func_id]))
    }

    pub fn add_reduce(&mut self, input: Id, func: Arc<ReduceFuncSame<S>>, dim: i32) -> Id {
        let func_id = Arc::as_ptr(&func) as *const () as usize;
        let func_name = func.as_str();
        self.reduce_funcs.insert(func_id, func);
        let graph_func_id = self
            .egraph
            .add(CpuBackendLanguage::ReduceFunc(FunctionLookup {
                id: func_id,
                func_type: func_name,
            }));
        let dim_id = self.egraph.add(CpuBackendLanguage::Dim(dim));
        self.egraph
            .add(CpuBackendLanguage::Reduce([input, graph_func_id, dim_id]))
    }

    pub fn add_broadcast(
        &mut self,
        lhs_input: Id,
        rhs_input: Id,
        func: Arc<BroadcastFuncSame<S>>,
        corresponding_dimensions: Vec<(i32, i32)>,
    ) -> Id {
        let func_id = Arc::as_ptr(&func) as *const () as usize;
        let func_name = func.as_str();
        self.broadcast_funcs.insert(func_id, func);
        let graph_func_id = self
            .egraph
            .add(CpuBackendLanguage::BroadcastFunc(FunctionLookup {
                id: func_id,
                func_type: func_name,
            }));
        let corr_dims_id = self.egraph.add(CpuBackendLanguage::CorrespondingDims(
            corresponding_dimensions.clone().into(),
        ));
        self.egraph.add(CpuBackendLanguage::Broadcast([
            lhs_input,
            rhs_input,
            graph_func_id,
            corr_dims_id,
        ]))
    }
}

impl<S: Storage> Default for CpuBackendContext<S> {
    fn default() -> Self {
        Self {
            tensors: HashMap::new(),
            map_funcs: HashMap::default(),
            reduce_funcs: HashMap::default(),
            broadcast_funcs: HashMap::default(),
            egraph: EGraph::default(),
            eval_node_id: None,
            rewrites: Vec::new(),
            executor: CpuExecutor,
        }
    }
}
