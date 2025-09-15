use core::{
    Layout,
    backends::{broadcast::BroadcastFuncSame, map::MapFuncSame, reduce::ReduceFuncSame},
    storage::Storage,
    tensor::{LazyTensor, Tensor},
};
use egg::{CostFunction, EGraph, Id, Language, Rewrite};
use std::{collections::HashMap, sync::Arc};

use crate::language::{CoreLanguage, FunctionLookup, TensorRef};

pub trait BackendContext: From<CoreContext<Self::BackendStorage>> {
    type BackendStorage: Storage;
    type BackendError;
    type BackendLanguage: Language + From<CoreLanguage>;

    fn add_rewrites(&mut self, rewrites: &[Rewrite<Self::BackendLanguage, ()>]);

    fn optimize(&mut self);

    fn evaluate<C: CostFunction<Self::BackendLanguage>>(
        &mut self,
        cost_func: C,
        node_id: Id,
    ) -> Result<Arc<Tensor<Self::BackendStorage>>, Self::BackendError>;
}

pub struct CoreContext<S: Storage> {
    pub tensors: HashMap<usize, Arc<Tensor<S>>>,
    pub map_funcs: HashMap<usize, Arc<MapFuncSame<S>>>,
    pub reduce_funcs: HashMap<usize, Arc<ReduceFuncSame<S>>>,
    pub broadcast_funcs: HashMap<usize, Arc<BroadcastFuncSame<S>>>,
    pub egraph: EGraph<CoreLanguage, ()>,
    pub history: Vec<(Id, CoreLanguage)>,
}

impl<S: Storage> CoreContext<S> {
    pub fn map_egraph<L: Language + From<CoreLanguage>>(&self) -> EGraph<L, ()> {
        let mut new_egraph = EGraph::new(());

        for (id, node) in &self.history {
            let new_id = new_egraph.add(node.clone().into());
            assert_eq!(*id, new_id);
        }

        new_egraph
    }

    pub fn last_added_id(&self) -> Option<Id> {
        self.history.last().map(|(id, _)| *id)
    }
}

impl<S: Storage> Default for CoreContext<S> {
    fn default() -> Self {
        Self {
            tensors: HashMap::new(),
            map_funcs: HashMap::default(),
            reduce_funcs: HashMap::default(),
            broadcast_funcs: HashMap::default(),
            egraph: EGraph::default(),
            history: Vec::new(),
        }
    }
}

impl<S: Storage> From<&LazyTensor<S>> for CoreContext<S> {
    fn from(lazy_tensor: &LazyTensor<S>) -> Self {
        let mut context_builder = CoreContextBulder::default();
        context_builder.build_recursively(lazy_tensor);
        context_builder.into()
    }
}

impl<S: Storage> From<CoreContextBulder<S>> for CoreContext<S> {
    fn from(builder: CoreContextBulder<S>) -> Self {
        builder.context
    }
}

pub struct CoreContextBulder<S: Storage> {
    context: CoreContext<S>,
}

impl<S: Storage> Default for CoreContextBulder<S> {
    fn default() -> Self {
        CoreContextBulder {
            context: CoreContext::default(),
        }
    }
}

impl<S: Storage> CoreContextBulder<S> {
    fn build_recursively(&mut self, tensor: &LazyTensor<S>) -> Id {
        match tensor {
            LazyTensor::Tensor(t) => self.build_tensor_expr(t.clone()),
            LazyTensor::Map {
                input,
                func,
                layout,
            } => self.build_map_expr(input, func.clone(), layout),
            LazyTensor::Reduce {
                input,
                dim,
                func,
                layout,
            } => self.build_reduce_expr(input, *dim, func.clone(), layout),
            LazyTensor::Broadcast {
                lhs_input,
                rhs_input,
                corresponding_dimensions,
                func,
                layout,
            } => self.build_broadcast_expr(
                lhs_input,
                rhs_input,
                corresponding_dimensions.clone(),
                func.clone(),
                layout,
            ),
        }
    }

    fn build_tensor_expr(&mut self, tensor: Arc<Tensor<S>>) -> Id {
        let tensor_id = Arc::as_ptr(&tensor) as usize;
        self.context.tensors.insert(tensor_id, tensor.clone());

        let tensor_ref = CoreLanguage::TensorRef(TensorRef { id: tensor_id });
        let tensor_ref_id = self.context.egraph.add(tensor_ref.clone());
        self.context
            .history
            .push((tensor_ref_id.clone(), tensor_ref));

        let shape = CoreLanguage::Shape((&tensor.layout).into());
        let shape_id = self.context.egraph.add(shape.clone());
        self.context.history.push((shape_id.clone(), shape));

        let tensor = CoreLanguage::Tensor([tensor_ref_id, shape_id]);
        let tensor_id = self.context.egraph.add(tensor.clone());
        self.context.history.push((tensor_id.clone(), tensor));

        tensor_id
    }

    fn build_map_expr(
        &mut self,
        input: &LazyTensor<S>,
        func: Arc<MapFuncSame<S>>,
        layout: &Layout,
    ) -> Id {
        let input_id = self.build_recursively(input);
        let func_id = Arc::as_ptr(&func) as *const () as usize;
        let func_name = func.as_str();
        self.context.map_funcs.insert(func_id, func);
        let map_func = CoreLanguage::MapFunc(FunctionLookup {
            id: func_id,
            func_type: func_name,
        });

        let graph_func_id = self.context.egraph.add(map_func.clone());
        self.context.history.push((graph_func_id.clone(), map_func));

        let shape = CoreLanguage::Shape(layout.into());
        let shape_id = self.context.egraph.add(shape.clone());
        self.context.history.push((shape_id.clone(), shape));

        let map = CoreLanguage::Map([input_id, graph_func_id, shape_id]);

        let map_id = self.context.egraph.add(map.clone());
        self.context.history.push((map_id.clone(), map));

        map_id
    }

    fn build_reduce_expr(
        &mut self,
        input: &LazyTensor<S>,
        dim: i32,
        func: Arc<ReduceFuncSame<S>>,
        layout: &Layout,
    ) -> Id {
        let input_id = self.build_recursively(input);
        let func_id = Arc::as_ptr(&func) as *const () as usize;
        let func_name = func.as_str();
        self.context.reduce_funcs.insert(func_id, func);

        let reduce_func = CoreLanguage::ReduceFunc(FunctionLookup {
            id: func_id,
            func_type: func_name,
        });

        let graph_func_id = self.context.egraph.add(reduce_func.clone());
        self.context
            .history
            .push((graph_func_id.clone(), reduce_func));

        let dim_node = CoreLanguage::Dim(dim);
        let dim_id = self.context.egraph.add(dim_node.clone());
        self.context.history.push((dim_id.clone(), dim_node));

        let shape = CoreLanguage::Shape(layout.into());
        let shape_id = self.context.egraph.add(shape.clone());
        self.context.history.push((shape_id.clone(), shape));

        let reduce = CoreLanguage::Reduce([input_id, graph_func_id, dim_id, shape_id]);
        let reduce_id = self.context.egraph.add(reduce.clone());
        self.context.history.push((reduce_id.clone(), reduce));
        reduce_id
    }

    fn build_broadcast_expr(
        &mut self,
        lhs_input: &LazyTensor<S>,
        rhs_input: &LazyTensor<S>,
        corresponding_dimensions: Vec<(i32, i32)>,
        func: Arc<BroadcastFuncSame<S>>,
        layout: &Layout,
    ) -> Id {
        let lhs_id = self.build_recursively(lhs_input);
        let rhs_id = self.build_recursively(rhs_input);
        let func_id = Arc::as_ptr(&func) as *const () as usize;
        let func_name = func.as_str();
        self.context.broadcast_funcs.insert(func_id, func);

        let broadcast_func = CoreLanguage::BroadcastFunc(FunctionLookup {
            id: func_id,
            func_type: func_name,
        });
        let graph_func_id = self.context.egraph.add(broadcast_func.clone());
        self.context
            .history
            .push((graph_func_id.clone(), broadcast_func));

        let corr_dims_node =
            CoreLanguage::CorrespondingDims(corresponding_dimensions.clone().into());
        let corr_dims_id = self.context.egraph.add(corr_dims_node.clone());
        self.context
            .history
            .push((corr_dims_id.clone(), corr_dims_node));

        let shape = CoreLanguage::Shape(layout.into());
        let shape_id = self.context.egraph.add(shape.clone());
        self.context.history.push((shape_id.clone(), shape));

        let broadcast =
            CoreLanguage::Broadcast([lhs_id, rhs_id, graph_func_id, corr_dims_id, shape_id]);
        let broadcast_id = self.context.egraph.add(broadcast.clone());
        self.context.history.push((broadcast_id.clone(), broadcast));

        broadcast_id
    }
}
