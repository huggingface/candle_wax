use egg::{CostFunction, Extractor, Id, Rewrite, Runner, define_language, rewrite};
use std::{collections::HashMap, sync::Arc};

use core::{
    Layout,
    backends::{
        Backend,
        map::{Map, MapFunc},
        reduce::{Reduce, ReduceFunc},
    },
    storage::Storage,
    tensor::{LazyTensor, Tensor},
};
use macros::BackendOps;

#[derive(BackendOps)]
pub struct CpuBackend {}

impl Backend for CpuBackend {
    fn eval<S: Storage>(tensor: LazyTensor<S>) -> Tensor<S> {
        let mut context = CpuBackendContext::from(tensor);

        context.eval_id = Some(
            context
                .egraph
                .add(CpuBackendLanguage::Output(context.eval_id.unwrap())),
        );

        context.add_rewrites();

        context.optimize();

        context.evaluate().as_ref().clone()
    }
}

define_language! {
    pub enum CpuBackendLanguage {
        "map" = Map([Id; 2]), // [input_expr, func_id]
        "reduce" = Reduce([Id; 3]), // [input_expr, dim, func_id]
        "fused_map_reduce" = FusedMapReduce([Id; 4]), // [input_expr, map_func_id, dim, reduce_func_id]
        "fused_reduce_map" = FusedReduceMap([Id; 4]), // [input_expr, reduce_func_id, dim, map_func_id]
        "output" = Output(Id), // input_expr

        Tensor(usize),
        MapFunc(usize),
        ReduceFunc(usize),
        Dim(i32),
    }
}

pub struct CpuBackendContext<S: Storage> {
    pub tensors: HashMap<usize, Arc<Tensor<S>>>,
    pub map_funcs: HashMap<usize, Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>>,
    pub reduce_funcs: HashMap<usize, Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>>,
    pub egraph: egg::EGraph<CpuBackendLanguage, ()>,
    pub eval_id: Option<Id>,
    pub rewrites: Vec<Rewrite<CpuBackendLanguage, ()>>,
}

impl<S: Storage> CpuBackendContext<S> {
    pub fn add_tensor(&mut self, tensor: Arc<Tensor<S>>) -> usize {
        let unique_id = Arc::as_ptr(&tensor) as usize;
        self.tensors.insert(unique_id, tensor);
        unique_id
    }

    pub fn add_map_func(&mut self, func: Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>) -> usize {
        let unique_id = Arc::as_ptr(&func) as *const () as usize;
        self.map_funcs.insert(unique_id, func);
        unique_id
    }

    pub fn add_reduce_func(
        &mut self,
        func: Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>,
    ) -> usize {
        let unique_id = Arc::as_ptr(&func) as *const () as usize;
        self.reduce_funcs.insert(unique_id, func);
        unique_id
    }

    pub fn add_rewrites(&mut self) {
        self.rewrites = vec![
            rewrite!("map-reduce-fusion";
                "(reduce (map ?x ?f) ?dim ?g)" =>
                "(fused_map_reduce ?x ?f ?dim ?g)"
            ),
            rewrite!("reduce-map-fusion";
                "(map (reduce ?x ?dim ?f) ?g)" =>
                "(fused_reduce_map ?x ?dim ?f ?g)"
            ),
        ];
    }

    pub fn optimize(&mut self) {
        let runner = Runner::default()
            .with_egraph(std::mem::take(&mut self.egraph))
            .run(&self.rewrites);

        self.egraph = runner.egraph;
    }

    pub fn evaluate(&mut self) -> Arc<Tensor<S>> {
        let extractor = Extractor::new(&self.egraph, CpuBackendCost);
        let eval_id = self.eval_id.expect("No evaluation ID set");
        let (cost, best_expr) = extractor.find_best(eval_id);

        println!("Best expression cost: {}", cost);

        self.execute_expression(&best_expr)
    }

    fn execute_expression(&self, expr: &egg::RecExpr<CpuBackendLanguage>) -> Arc<Tensor<S>> {
        let root = expr.last().expect("Expression is empty");
        self.execute_node(root, expr)
    }

    fn execute_node(
        &self,
        node: &CpuBackendLanguage,
        expr: &egg::RecExpr<CpuBackendLanguage>,
    ) -> Arc<Tensor<S>> {
        match node {
            CpuBackendLanguage::Tensor(tensor_id) => {
                let tensor = self
                    .tensors
                    .get(tensor_id)
                    .expect("Tensor not found in context");
                tensor.clone()
            }

            CpuBackendLanguage::Map([input_id, func_id]) => {
                let input = self.execute_node(&expr[*input_id], expr);
                let func_lookup = match &expr[*func_id] {
                    CpuBackendLanguage::MapFunc(f) => f,
                    _ => panic!("Expected map func"),
                };

                let func = self
                    .map_funcs
                    .get(func_lookup)
                    .expect("Map function not found in context");

                Arc::new(Tensor::new(
                    input.layout.clone(),
                    func.call(&input.layout, &input.storage),
                ))
            }

            CpuBackendLanguage::Reduce([input_id, dim_id, func_id]) => {
                let input = self.execute_node(&expr[*input_id], expr);
                let dim = match &expr[*dim_id] {
                    CpuBackendLanguage::Dim(d) => *d,
                    _ => panic!("Expected dimension"),
                };
                let func_lookup = match &expr[*func_id] {
                    CpuBackendLanguage::ReduceFunc(f) => f,
                    _ => panic!("Expected reduce func"),
                };
                let func = self
                    .reduce_funcs
                    .get(func_lookup)
                    .expect("Reduce function not found in context");

                Arc::new(Tensor::new(
                    input.layout.clone(),
                    func.call(&input.layout, &input.storage, dim),
                ))
            }

            CpuBackendLanguage::FusedMapReduce([input_id, map_func_id, dim_id, reduce_func_id]) => {
                let input = self.execute_node(&expr[*input_id], expr);

                let map_func_lookup = match &expr[*map_func_id] {
                    CpuBackendLanguage::MapFunc(f) => f,
                    _ => panic!("Expected map func"),
                };
                let map_func = self
                    .map_funcs
                    .get(map_func_lookup)
                    .expect("Map function not found in context");

                let reduce_func_lookup = match &expr[*reduce_func_id] {
                    CpuBackendLanguage::ReduceFunc(f) => f,
                    _ => panic!("Expected reduce func"),
                };
                let reduce_func = self
                    .reduce_funcs
                    .get(reduce_func_lookup)
                    .expect("Reduce function not found in context");
                let dim = match &expr[*dim_id] {
                    CpuBackendLanguage::Dim(d) => *d,
                    _ => panic!("Expected dimension"),
                };

                self.execute_fused_map_reduce(&input, map_func, dim, reduce_func)
            }

            CpuBackendLanguage::FusedReduceMap([input_id, dim_id, reduce_func_id, map_func_id]) => {
                println!("Executing fused reduce-map operation");
                let input = self.execute_node(&expr[*input_id], expr);

                let map_func_lookup = match &expr[*map_func_id] {
                    CpuBackendLanguage::MapFunc(f) => f,
                    _ => panic!("Expected map func"),
                };
                let map_func = self
                    .map_funcs
                    .get(map_func_lookup)
                    .expect("Map function not found in context");

                let reduce_func_lookup = match &expr[*reduce_func_id] {
                    CpuBackendLanguage::ReduceFunc(f) => f,
                    _ => panic!("Expected reduce func"),
                };
                let reduce_func = self
                    .reduce_funcs
                    .get(reduce_func_lookup)
                    .expect("Reduce function not found in context");
                let dim = match &expr[*dim_id] {
                    CpuBackendLanguage::Dim(d) => *d,
                    _ => panic!("Expected dimension"),
                };

                self.execute_fused_reduce_map(&input, dim, reduce_func, map_func)
            }

            CpuBackendLanguage::Dim(_) => {
                panic!("Cannot execute dimension node directly")
            }

            CpuBackendLanguage::MapFunc(_) => {
                panic!("Cannot execute map func node directly")
            }

            CpuBackendLanguage::ReduceFunc(_) => {
                panic!("Cannot execute reduce func node directly")
            }

            CpuBackendLanguage::Output(input_id) => self.execute_node(&expr[*input_id], expr),
        }
    }

    fn execute_fused_map_reduce(
        &self,
        input: &Arc<Tensor<S>>,
        map_func: &Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>,
        dim: i32,
        reduce_func: &Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>,
    ) -> Arc<Tensor<S>> {
        let mapped = map_func.call(&input.layout, &input.storage);
        Arc::new(Tensor::new(
            input
                .layout
                .reduce(input.layout.signed_dim_to_unsigned_dim(dim)),
            reduce_func.call(&input.layout, &mapped, dim),
        ))
    }

    fn execute_fused_reduce_map(
        &self,
        input: &Arc<Tensor<S>>,
        dim: i32,
        reduce_func: &Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>,
        map_func: &Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>,
    ) -> Arc<Tensor<S>> {
        let reduced = reduce_func.call(&input.layout, &input.storage, dim);
        Arc::new(Tensor::new(
            input
                .layout
                .reduce(input.layout.signed_dim_to_unsigned_dim(dim)),
            map_func.call(&input.layout, &reduced),
        ))
    }
}

impl<S: Storage> Default for CpuBackendContext<S> {
    fn default() -> Self {
        Self {
            tensors: HashMap::new(),
            map_funcs: HashMap::new(),
            reduce_funcs: HashMap::new(),
            egraph: egg::EGraph::default(),
            eval_id: None,
            rewrites: Vec::new(),
        }
    }
}

impl<S: Storage> From<LazyTensor<S>> for CpuBackendContext<S> {
    fn from(tensor: LazyTensor<S>) -> Self {
        match tensor {
            LazyTensor::Tensor(t) => {
                let mut context = CpuBackendContext::default();
                let tensor_id = context.add_tensor(t);
                let id = context.egraph.add(CpuBackendLanguage::Tensor(tensor_id));
                context.eval_id = Some(id);
                context
            }
            LazyTensor::Map { input, func } => {
                let mut context = CpuBackendContext::from(*input);
                let func_id = context.add_map_func(func);
                let map_func_id = context.egraph.add(CpuBackendLanguage::MapFunc(func_id));
                let id = context.egraph.add(CpuBackendLanguage::Map([
                    context.eval_id.unwrap(),
                    map_func_id,
                ]));
                context.eval_id = Some(id);
                context
            }
            LazyTensor::Reduce { input, dim, func } => {
                let mut context = CpuBackendContext::from(*input);
                let func_id = context.add_reduce_func(func);
                let reduce_func_id = context.egraph.add(CpuBackendLanguage::ReduceFunc(func_id));
                let dim_id = context.egraph.add(CpuBackendLanguage::Dim(dim));
                let id: Id = context.egraph.add(CpuBackendLanguage::Reduce([
                    context.eval_id.unwrap(),
                    dim_id,
                    reduce_func_id,
                ]));
                context.eval_id = Some(id);
                context
            }
        }
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
            CpuBackendLanguage::MapFunc(_) => 0,
            CpuBackendLanguage::ReduceFunc(_) => 0,
            CpuBackendLanguage::Tensor(_) => 0,
            CpuBackendLanguage::Map(args) => costs(args[0]) + 10,
            CpuBackendLanguage::Reduce(args) => costs(args[0]) + 15,
            CpuBackendLanguage::FusedMapReduce(args) => costs(args[0]) + 20,
            CpuBackendLanguage::FusedReduceMap(args) => costs(args[0]) + 20,
            CpuBackendLanguage::Dim(_) => 0,
            CpuBackendLanguage::Output(id) => costs(*id),
        }
    }
}
