use egg::{
    CostFunction, EGraph, Extractor, Id, Rewrite, Runner, Subst, Var, define_language, rewrite,
};
use regex::Regex;
use std::{collections::HashMap, sync::Arc};

use core::{
    Layout,
    backends::{
        Backend, LazyBackend,
        broadcast::{Broadcast, BroadcastFunc},
        map::{Map, MapFunc},
        reduce::{Reduce, ReduceFunc},
    },
    storage::Storage,
    tensor::{LazyTensor, Tensor},
};
use macros::BackendOps;

#[derive(Debug)]
pub enum CpuBackendError {
    TensorNotFound(usize),

    FunctionNotFound(String),

    UnexpectedNodeType { expected: String, found: String },

    CorrespondingDimensionsNotFound(usize),

    NoEvaluationId,

    EmptyExpression,
}

pub type Result<T> = std::result::Result<T, CpuBackendError>;

#[derive(BackendOps)]
pub struct CpuBackend {}

impl Backend for CpuBackend {}

impl LazyBackend for CpuBackend {
    fn eval<S: Storage>(tensor: LazyTensor<S>) -> Tensor<S> {
        let mut context = CpuBackendContext::from(tensor);

        context.eval_id = Some(
            context
                .egraph
                .add(CpuBackendLanguage::Output(context.eval_id.unwrap())),
        );

        context.add_rewrites();
        context.optimize();

        match context.evaluate() {
            Ok(result) => result.as_ref().clone(),
            Err(_e) => panic!("Evaluation failed"),
        }
    }
}

define_language! {
    pub enum CpuBackendLanguage {
        "map" = Map([Id; 2]),        // [input_expr, func_id]
        "reduce" = Reduce([Id; 3]),  // [input_expr, dim, func_id]
        "broadcast" = Broadcast([Id; 4]), // [lhs_input_expr, rhs_input_expr, corresponding_dims, func_id]

        "fused_map_reduce" = FusedMapReduce([Id; 4]), // [input_expr, map_func_id, dim, reduce_func_id]
        "fused_reduce_map" = FusedReduceMap([Id; 4]), // [input_expr, reduce_func_id, dim, map_func_id]
        "fused_matmul" = FusedMatmul([Id; 4]),        // [lhs_input_expr, rhs_input_expr, corresponding_dims, dim]

        "output" = Output(Id), // input_expr

        MapFunc(String),
        ReduceFunc(String),
        BroadcastFunc(String),

        Tensor(usize),
        CorrespondingDims(usize),
        Dim(i32),
    }
}

pub trait BackendContext: Default {
    type S: Storage;

    fn set_eval_id(&mut self, id: Id);
    fn get_eval_id(&self) -> Option<Id>;

    fn add_tensor(&mut self, tensor: Arc<Tensor<Self::S>>) -> Id;
    fn add_map(
        &mut self,
        input: Id,
        func: Arc<
            dyn MapFunc<Self::S, Self::S, <Self::S as Storage>::Inner, <Self::S as Storage>::Inner>,
        >,
    ) -> Id;
    fn add_reduce(
        &mut self,
        input: Id,
        dim: i32,
        func: Arc<
            dyn ReduceFunc<
                    Self::S,
                    Self::S,
                    <Self::S as Storage>::Inner,
                    <Self::S as Storage>::Inner,
                >,
        >,
    ) -> Id;
    fn add_broadcast(
        &mut self,
        lhs_input: Id,
        rhs_input: Id,
        corresponding_dimensions: Vec<(i32, i32)>,
        func: Arc<
            dyn BroadcastFunc<
                    Self::S,
                    Self::S,
                    Self::S,
                    <Self::S as Storage>::Inner,
                    <Self::S as Storage>::Inner,
                    <Self::S as Storage>::Inner,
                >,
        >,
    ) -> Id;
}

pub struct ExpressionBuilder<C: BackendContext> {
    context: C,
}

impl<C: BackendContext> ExpressionBuilder<C> {
    pub fn new() -> Self {
        Self {
            context: C::default(),
        }
    }

    pub fn build_from_lazy_tensor(mut self, tensor: &LazyTensor<C::S>) -> C {
        let eval_id = self.build_expression(tensor);
        self.context.set_eval_id(eval_id);
        self.context
    }

    fn build_expression(&mut self, tensor: &LazyTensor<C::S>) -> Id {
        match tensor {
            LazyTensor::Tensor(t) => self.build_tensor_expr(t.clone()),
            LazyTensor::Map { input, func } => self.build_map_expr(input, func.clone()),
            LazyTensor::Reduce { input, dim, func } => {
                self.build_reduce_expr(input, *dim, func.clone())
            }
            LazyTensor::Broadcast {
                lhs_input,
                rhs_input,
                corresponding_dimensions,
                func,
            } => self.build_broadcast_expr(
                lhs_input,
                rhs_input,
                corresponding_dimensions.clone(),
                func.clone(),
            ),
        }
    }

    fn build_tensor_expr(&mut self, tensor: Arc<Tensor<C::S>>) -> Id {
        // let tensor_id = self.context.add_tensor(tensor.clone());
        // self.context
        //     .egraph
        //     .add(CpuBackendLanguage::Tensor(tensor_id))
        self.context.add_tensor(tensor)
    }

    fn build_map_expr(
        &mut self,
        input: &LazyTensor<C::S>,
        func: Arc<dyn MapFunc<C::S, C::S, <C::S as Storage>::Inner, <C::S as Storage>::Inner>>,
    ) -> Id {
        let input_id = self.build_expression(input);
        self.context.add_map(input_id, func)
        // let func_name = self.context.add_map_func(func.clone());
        // let func_id = self
        //     .context
        //     .egraph
        //     .add(CpuBackendLanguage::MapFunc(func_name));
        // self.context
        //     .egraph
        //     .add(CpuBackendLanguage::Map([input_id, func_id]))
    }

    fn build_reduce_expr(
        &mut self,
        input: &LazyTensor<C::S>,
        dim: i32,
        func: Arc<dyn ReduceFunc<C::S, C::S, <C::S as Storage>::Inner, <C::S as Storage>::Inner>>,
    ) -> Id {
        let input_id = self.build_expression(input);
        self.context.add_reduce(input_id, dim, func)
        // let func_name = self.context.add_reduce_func(func.clone());
        // let func_id = self
        //     .context
        //     .egraph
        //     .add(CpuBackendLanguage::ReduceFunc(func_name));
        // let dim_id = self.context.egraph.add(CpuBackendLanguage::Dim(dim));
        // self.context
        //     .egraph
        //     .add(CpuBackendLanguage::Reduce([input_id, dim_id, func_id]))
    }

    fn build_broadcast_expr(
        &mut self,
        lhs_input: &LazyTensor<C::S>,
        rhs_input: &LazyTensor<C::S>,
        corresponding_dimensions: Vec<(i32, i32)>,
        func: Arc<
            dyn BroadcastFunc<
                    C::S,
                    C::S,
                    C::S,
                    <C::S as Storage>::Inner,
                    <C::S as Storage>::Inner,
                    <C::S as Storage>::Inner,
                >,
        >,
    ) -> Id {
        let lhs_id = self.build_expression(lhs_input);
        let rhs_id = self.build_expression(rhs_input);
        self.context
            .add_broadcast(lhs_id, rhs_id, corresponding_dimensions, func)
        // let func_name = self.context.add_broadcast_func(func.clone());
        // let func_id = self
        //     .context
        //     .egraph
        //     .add(CpuBackendLanguage::BroadcastFunc(func_name));
        // let dims_id = self
        //     .context
        //     .add_corresponding_dims(corresponding_dimensions.clone());
        // let corrdim_id = self
        //     .context
        //     .egraph
        //     .add(CpuBackendLanguage::CorrespondingDims(dims_id));

        // self.context.egraph.add(CpuBackendLanguage::Broadcast([
        //     lhs_id, rhs_id, corrdim_id, func_id,
        // ]))
    }
}

pub trait CpuNodeExecutor<S: Storage> {
    fn execute_node(
        &self,
        node: &CpuBackendLanguage,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>>;
}

pub struct CpuExecutor;

impl<S: Storage> CpuNodeExecutor<S> for CpuExecutor {
    fn execute_node(
        &self,
        node: &CpuBackendLanguage,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>> {
        match node {
            CpuBackendLanguage::Tensor(tensor_id) => Ok(context.get_tensor(*tensor_id)?.clone()),

            CpuBackendLanguage::Map([input_id, func_id]) => {
                self.execute_map(*input_id, *func_id, expr, context)
            }

            CpuBackendLanguage::Reduce([input_id, dim_id, func_id]) => {
                self.execute_reduce(*input_id, *dim_id, *func_id, expr, context)
            }

            CpuBackendLanguage::Broadcast([lhs_id, rhs_id, corrdims_id, func_id]) => {
                self.execute_broadcast(*lhs_id, *rhs_id, *corrdims_id, *func_id, expr, context)
            }

            CpuBackendLanguage::FusedMapReduce([input_id, map_func_id, dim_id, reduce_func_id]) => {
                self.execute_fused_map_reduce(
                    *input_id,
                    *map_func_id,
                    *dim_id,
                    *reduce_func_id,
                    expr,
                    context,
                )
            }

            CpuBackendLanguage::FusedReduceMap([input_id, dim_id, reduce_func_id, map_func_id]) => {
                self.execute_fused_reduce_map(
                    *input_id,
                    *dim_id,
                    *reduce_func_id,
                    *map_func_id,
                    expr,
                    context,
                )
            }

            CpuBackendLanguage::FusedMatmul([lhs_id, rhs_id, corrdims_id, dim_id]) => {
                self.execute_fused_matmul(*lhs_id, *rhs_id, *corrdims_id, *dim_id, expr, context)
            }

            CpuBackendLanguage::Output(input_id) => {
                self.execute_node(&expr[*input_id], expr, context)
            }

            _ => Err(CpuBackendError::UnexpectedNodeType {
                expected: "executable node".to_string(),
                found: format!("{:?}", node),
            }),
        }
    }
}

impl CpuExecutor {
    fn execute_map<S: Storage>(
        &self,
        input_id: Id,
        func_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>> {
        let input = self.execute_node(&expr[input_id], expr, context)?;

        let func_lookup = match &expr[func_id] {
            CpuBackendLanguage::MapFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "MapFunc".to_string(),
                    found: format!("{:?}", &expr[func_id]),
                });
            }
        };

        let func = context.get_map_func(func_lookup)?;

        Ok(Arc::new(Tensor::new(
            input.layout.clone(),
            func.call(&input.layout, &input.storage),
        )))
    }

    fn execute_reduce<S: Storage>(
        &self,
        input_id: Id,
        dim_id: Id,
        func_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>> {
        let input = self.execute_node(&expr[input_id], expr, context)?;

        let dim = match &expr[dim_id] {
            CpuBackendLanguage::Dim(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "Dim".to_string(),
                    found: format!("{:?}", &expr[dim_id]),
                });
            }
        };

        let func_lookup = match &expr[func_id] {
            CpuBackendLanguage::ReduceFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "ReduceFunc".to_string(),
                    found: format!("{:?}", &expr[func_id]),
                });
            }
        };

        let func = context.get_reduce_func(func_lookup)?;

        Ok(Arc::new(Tensor::new(
            input.layout.clone(),
            func.call(&input.layout, &input.storage, dim),
        )))
    }

    fn execute_broadcast<S: Storage>(
        &self,
        lhs_id: Id,
        rhs_id: Id,
        corrdims_id: Id,
        func_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>> {
        let lhs_input = self.execute_node(&expr[lhs_id], expr, context)?;
        let rhs_input = self.execute_node(&expr[rhs_id], expr, context)?;

        let corrdims_lookup = match &expr[corrdims_id] {
            CpuBackendLanguage::CorrespondingDims(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "CorrespondingDims".to_string(),
                    found: format!("{:?}", &expr[corrdims_id]),
                });
            }
        };

        let corrdims = context.get_corresponding_dims(corrdims_lookup)?;

        let func_lookup = match &expr[func_id] {
            CpuBackendLanguage::BroadcastFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType { 
                    expected: "BroadcastFunc".to_string(),
                    found: format!("{:?}", &expr[func_id]),
                });
            }
        };

        let func = context.get_broadcast_func(func_lookup)?;

        Ok(Arc::new(Tensor::new(
            lhs_input.layout.broadcast(
                &rhs_input.layout,
                &lhs_input
                    .layout
                    .signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
                        &rhs_input.layout,
                        corrdims,
                    ),
            ),
            func.call(
                &lhs_input.layout,
                &lhs_input.storage,
                &rhs_input.layout,
                &rhs_input.storage,
                corrdims,
            ),
        )))
    }

    fn execute_fused_map_reduce<S: Storage>(
        &self,
        input_id: Id,
        map_func_id: Id,
        dim_id: Id,
        reduce_func_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>> {
        let input = self.execute_node(&expr[input_id], expr, context)?;

        let map_func_lookup = match &expr[map_func_id] {
            CpuBackendLanguage::MapFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "MapFunc".to_string(),
                    found: format!("{:?}", &expr[map_func_id]),
                });
            }
        };

        let reduce_func_lookup = match &expr[reduce_func_id] {
            CpuBackendLanguage::ReduceFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "ReduceFunc".to_string(),
                    found: format!("{:?}", &expr[reduce_func_id]),
                });
            }
        };

        let dim = match &expr[dim_id] {
            CpuBackendLanguage::Dim(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "Dim".to_string(),
                    found: format!("{:?}", &expr[dim_id]),
                });
            }
        };

        let map_func = context.get_map_func(map_func_lookup)?;
        let reduce_func = context.get_reduce_func(reduce_func_lookup)?;

        let mapped = map_func.call(&input.layout, &input.storage);
        Ok(Arc::new(Tensor::new(
            input
                .layout
                .reduce(input.layout.signed_dim_to_unsigned_dim(dim)),
            reduce_func.call(&input.layout, &mapped, dim),
        )))
    }

    fn execute_fused_reduce_map<S: Storage>(
        &self,
        input_id: Id,
        dim_id: Id,
        reduce_func_id: Id,
        map_func_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>> {
        println!("Executing fused reduce-map operation");
        let input = self.execute_node(&expr[input_id], expr, context)?;

        let map_func_lookup = match &expr[map_func_id] {
            CpuBackendLanguage::MapFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "MapFunc".to_string(),
                    found: format!("{:?}", &expr[map_func_id]),
                });
            }
        };

        let reduce_func_lookup = match &expr[reduce_func_id] {
            CpuBackendLanguage::ReduceFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "ReduceFunc".to_string(),
                    found: format!("{:?}", &expr[reduce_func_id]),
                });
            }
        };

        let dim = match &expr[dim_id] {
            CpuBackendLanguage::Dim(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "Dim".to_string(),
                    found: format!("{:?}", &expr[dim_id]),
                });
            }
        };

        let map_func = context.get_map_func(map_func_lookup)?;
        let reduce_func = context.get_reduce_func(reduce_func_lookup)?;

        let reduced = reduce_func.call(&input.layout, &input.storage, dim);
        Ok(Arc::new(Tensor::new(
            input
                .layout
                .reduce(input.layout.signed_dim_to_unsigned_dim(dim)),
            map_func.call(&input.layout, &reduced),
        )))
    }

    fn execute_fused_matmul<S: Storage>(
        &self,
        lhs_id: Id,
        rhs_id: Id,
        corrdims_id: Id,
        dim_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>> {
        println!("Executing fused matmul operation");
        let lhs_input = self.execute_node(&expr[lhs_id], expr, context)?;
        let rhs_input = self.execute_node(&expr[rhs_id], expr, context)?;

        let corrdims_lookup = match &expr[corrdims_id] {
            CpuBackendLanguage::CorrespondingDims(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "CorrespondingDims".to_string(),
                    found: format!("{:?}", &expr[corrdims_id]),
                });
            }
        };

        let corrdims = context.get_corresponding_dims(corrdims_lookup)?;

        let dim = match &expr[dim_id] {
            CpuBackendLanguage::Dim(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "Dim".to_string(),
                    found: format!("{:?}", &expr[dim_id]),
                });
            }
        };

        let multiply_func = context.get_broadcast_func(&format!(
            "CpuMultiply({}, {} -> {})",
            std::any::type_name::<S>(),
            std::any::type_name::<S>(),
            std::any::type_name::<S>()
        ))?;

        let sum_func = context.get_reduce_func(&format!(
            "CpuSum({} -> {})",
            std::any::type_name::<S>(),
            std::any::type_name::<S>()
        ))?;

        let broadcasted = multiply_func.call(
            &lhs_input.layout,
            &lhs_input.storage,
            &rhs_input.layout,
            &rhs_input.storage,
            corrdims,
        );

        let broadcasted_layout = lhs_input.layout.broadcast(
            &rhs_input.layout,
            &lhs_input
                .layout
                .signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
                    &rhs_input.layout,
                    corrdims,
                ),
        );

        let reduced_storage = sum_func.call(&broadcasted_layout, &broadcasted, dim);

        Ok(Arc::new(Tensor::new(
            broadcasted_layout.reduce(broadcasted_layout.signed_dim_to_unsigned_dim(dim)),
            reduced_storage,
        )))
    }
}

pub struct CpuBackendContext<S: Storage> {
    pub tensors: HashMap<usize, Arc<Tensor<S>>>,
    pub map_funcs: HashMap<String, Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>>,
    pub reduce_funcs: HashMap<String, Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>>,
    pub broadcast_funcs:
        HashMap<String, Arc<dyn BroadcastFunc<S, S, S, S::Inner, S::Inner, S::Inner>>>,
    pub corresponding_dims: HashMap<usize, Vec<(i32, i32)>>,
    pub egraph: EGraph<CpuBackendLanguage, ()>,
    pub eval_id: Option<Id>,
    pub rewrites: Vec<Rewrite<CpuBackendLanguage, ()>>,
    pub executor: CpuExecutor,
}

impl<S: Storage> CpuBackendContext<S> {
    pub fn add_tensor(&mut self, tensor: Arc<Tensor<S>>) -> usize {
        let unique_id = Arc::as_ptr(&tensor) as usize;
        self.tensors.insert(unique_id, tensor);
        unique_id
    }

    pub fn get_tensor(&self, tensor_id: usize) -> Result<&Arc<Tensor<S>>> {
        self.tensors
            .get(&tensor_id)
            .ok_or(CpuBackendError::TensorNotFound(tensor_id))
    }

    pub fn add_map_func(&mut self, func: Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>) -> String {
        let as_str = func.as_str();
        self.map_funcs.insert(as_str.clone(), func);
        as_str
    }

    pub fn get_map_func(
        &self,
        func_name: &str,
    ) -> Result<&Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>> {
        self.map_funcs
            .get(func_name)
            .ok_or_else(|| CpuBackendError::FunctionNotFound(func_name.to_string()))
    }

    pub fn add_reduce_func(
        &mut self,
        func: Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>,
    ) -> String {
        let as_str = func.as_str();
        self.reduce_funcs.insert(as_str.clone(), func);
        as_str
    }

    pub fn get_reduce_func(
        &self,
        func_name: &str,
    ) -> Result<&Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>> {
        self.reduce_funcs
            .get(func_name)
            .ok_or_else(|| CpuBackendError::FunctionNotFound(func_name.to_string()))
    }

    pub fn add_broadcast_func(
        &mut self,
        func: Arc<dyn BroadcastFunc<S, S, S, S::Inner, S::Inner, S::Inner>>,
    ) -> String {
        let as_str = func.as_str();
        self.broadcast_funcs.insert(as_str.clone(), func);
        as_str
    }

    pub fn get_broadcast_func(
        &self,
        func_name: &str,
    ) -> Result<&Arc<dyn BroadcastFunc<S, S, S, S::Inner, S::Inner, S::Inner>>> {
        self.broadcast_funcs
            .get(func_name)
            .ok_or_else(|| CpuBackendError::FunctionNotFound(func_name.to_string()))
    }

    pub fn add_corresponding_dims(&mut self, dims: Vec<(i32, i32)>) -> usize {
        let unique_id = &dims as *const _ as usize;
        self.corresponding_dims.insert(unique_id, dims);
        unique_id
    }

    pub fn get_corresponding_dims(&self, dims_id: usize) -> Result<&Vec<(i32, i32)>> {
        self.corresponding_dims
            .get(&dims_id)
            .ok_or(CpuBackendError::CorrespondingDimensionsNotFound(dims_id))
    }

    pub fn add_rewrites(&mut self) {
        self.rewrites.extend([
            rewrite!("map-reduce-fusion";
                "(reduce (map ?x ?f) ?dim ?g)" =>
                "(fused_map_reduce ?x ?f ?dim ?g)"
            ),
            rewrite!("reduce-map-fusion";
                "(map (reduce ?x ?dim ?f) ?g)" =>
                "(fused_reduce_map ?x ?dim ?f ?g)"
            ),
            rewrite!("fused-matmul";
                "(reduce (broadcast ?x ?y ?corrdims ?multiply_func) ?dim ?sum_func)" =>
                "(fused_matmul ?x ?y ?corrdims ?dim)"
                if is_matmul("?multiply_func".parse().unwrap(), "?sum_func".parse().unwrap())
            ),
        ])
    }

    pub fn optimize(&mut self) {
        let runner = Runner::default()
            .with_egraph(std::mem::take(&mut self.egraph))
            .run(&self.rewrites);

        self.egraph = runner.egraph;
    }

    pub fn evaluate(&mut self) -> Result<Arc<Tensor<S>>> {
        let extractor = Extractor::new(&self.egraph, CpuBackendCost);
        let eval_id = self.eval_id.ok_or(CpuBackendError::NoEvaluationId)?;
        let (cost, best_expr) = extractor.find_best(eval_id);

        println!("Best expression cost: {}", cost);
        let root = best_expr.last().ok_or(CpuBackendError::EmptyExpression)?;
        self.executor.execute_node(root, &best_expr, self)
    }
}

impl<S: Storage> BackendContext for CpuBackendContext<S> {
    type S = S;

    fn set_eval_id(&mut self, id: Id) {
        self.eval_id = Some(id);
    }

    fn get_eval_id(&self) -> Option<Id> {
        self.eval_id
    }

    fn add_tensor(&mut self, tensor: Arc<Tensor<Self::S>>) -> Id {
        let tensor_name = self.add_tensor(tensor);
        self.egraph.add(CpuBackendLanguage::Tensor(tensor_name))
    }

    fn add_map(
        &mut self,
        input: Id,
        func: Arc<
            dyn MapFunc<Self::S, Self::S, <Self::S as Storage>::Inner, <Self::S as Storage>::Inner>,
        >,
    ) -> Id {
        let func_name = self.add_map_func(func);
        let func_id = self.egraph.add(CpuBackendLanguage::MapFunc(func_name));
        self.egraph.add(CpuBackendLanguage::Map([input, func_id]))
    }

    fn add_reduce(
        &mut self,
        input: Id,
        dim: i32,
        func: Arc<
            dyn ReduceFunc<
                    Self::S,
                    Self::S,
                    <Self::S as Storage>::Inner,
                    <Self::S as Storage>::Inner,
                >,
        >,
    ) -> Id {
        let func_name = self.add_reduce_func(func);
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
                    Self::S,
                    Self::S,
                    Self::S,
                    <Self::S as Storage>::Inner,
                    <Self::S as Storage>::Inner,
                    <Self::S as Storage>::Inner,
                >,
        >,
    ) -> Id {
        let func_name = self.add_broadcast_func(func);
        let func_id = self
            .egraph
            .add(CpuBackendLanguage::BroadcastFunc(func_name));
        let lhs_dims = self.add_corresponding_dims(corresponding_dimensions);
        let dims_id = self
            .egraph
            .add(CpuBackendLanguage::CorrespondingDims(lhs_dims));
        self.egraph.add(CpuBackendLanguage::Broadcast([
            lhs_input, rhs_input, dims_id, func_id,
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
            corresponding_dims: HashMap::new(),
            egraph: EGraph::default(),
            eval_id: None,
            rewrites: Vec::new(),
            executor: CpuExecutor,
        }
    }
}

impl<S: Storage> From<LazyTensor<S>> for CpuBackendContext<S> {
    fn from(tensor: LazyTensor<S>) -> Self {
        ExpressionBuilder::new().build_from_lazy_tensor(&tensor)
    }
}

#[derive(Default)]
struct CpuBackendCost;

impl CostFunction<CpuBackendLanguage> for CpuBackendCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &CpuBackendLanguage, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            // Leaf nodes have no cost
            CpuBackendLanguage::MapFunc(_) => 0,
            CpuBackendLanguage::ReduceFunc(_) => 0,
            CpuBackendLanguage::BroadcastFunc(_) => 0,
            CpuBackendLanguage::Tensor(_) => 0,
            CpuBackendLanguage::Dim(_) => 0,
            CpuBackendLanguage::CorrespondingDims(_) => 0,

            // Basic operations
            CpuBackendLanguage::Map(args) => costs(args[0]) + 10,
            CpuBackendLanguage::Reduce(args) => costs(args[0]) + 15,
            CpuBackendLanguage::Broadcast(args) => costs(args[0]) + costs(args[1]) + 25,

            // Fused operations (more efficient)
            CpuBackendLanguage::FusedMapReduce(args) => costs(args[0]) + 20,
            CpuBackendLanguage::FusedReduceMap(args) => costs(args[0]) + 20,
            CpuBackendLanguage::FusedMatmul(args) => costs(args[0]) + costs(args[1]) + 30,

            // Output wrapper
            CpuBackendLanguage::Output(id) => costs(*id),
        }
    }
}

fn is_matmul(f1: Var, f2: Var) -> impl Fn(&mut EGraph<CpuBackendLanguage, ()>, Id, &Subst) -> bool {
    let mul_pattern = Regex::new(r".*Multiply.*").unwrap();
    let sum_pattern = Regex::new(r".*Sum.*").unwrap();

    move |egraph, _, subst| {
        let f1_matches = egraph[subst[f1]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::BroadcastFunc(func_name) = node {
                mul_pattern.is_match(func_name)
            } else {
                false
            }
        });

        let f2_matches = egraph[subst[f2]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::ReduceFunc(func_name) = node {
                sum_pattern.is_match(func_name)
            } else {
                false
            }
        });

        f1_matches && f2_matches
    }
}
