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

        context.evaluate().as_ref().clone()
    }
}

define_language! {
    pub enum CpuBackendLanguage {
        "map" = Map([Id; 2]), // [input_expr, func_id]
        "reduce" = Reduce([Id; 3]), // [input_expr, dim, func_id]
        "broadcast" = Broadcast([Id; 4]), // [lhs_input_expr, rhs_input_expr, corresponding_dims, func_id]
        "fused_map_reduce" = FusedMapReduce([Id; 4]), // [input_expr, map_func_id, dim, reduce_func_id]
        "fused_reduce_map" = FusedReduceMap([Id; 4]), // [input_expr, reduce_func_id, dim, map_func_id]
        "fused_matmul" = FusedMatmul([Id; 4]), // [lhs_input_expr, rhs_input_expr, corresponding_dims, dim]
        "output" = Output(Id), // input_expr

        MapFunc(String),
        ReduceFunc(String),
        BroadcastFunc(String),

        Tensor(usize),
        CorrespondingDims(usize),
        Dim(i32),
    }
}

fn is_matmul(f1: Var, f2: Var) -> impl Fn(&mut EGraph<CpuBackendLanguage, ()>, Id, &Subst) -> bool {
    let mul_pattern = Regex::new(r".*Multiply.*").unwrap();
    let sum_pattern = Regex::new(r".*Sum.*").unwrap();

    move |egraph, _, subst| {
        // Check if any node in the equivalence class matches the patterns
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

pub struct CpuBackendContext<S: Storage> {
    pub tensors: HashMap<usize, Arc<Tensor<S>>>,
    pub map_funcs: HashMap<String, Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>>,
    pub reduce_funcs: HashMap<String, Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>>,
    pub broadcast_funcs:
        HashMap<String, Arc<dyn BroadcastFunc<S, S, S, S::Inner, S::Inner, S::Inner>>>,
    pub corresponding_dims: HashMap<usize, Vec<(i32, i32)>>,
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

    pub fn add_map_func(&mut self, func: Arc<dyn MapFunc<S, S, S::Inner, S::Inner>>) -> String {
        let fstr = func.as_str();
        self.map_funcs.insert(fstr.clone(), func);
        fstr
    }

    pub fn add_reduce_func(
        &mut self,
        func: Arc<dyn ReduceFunc<S, S, S::Inner, S::Inner>>,
    ) -> String {
        let fstr = func.as_str();
        self.reduce_funcs.insert(fstr.clone(), func);
        fstr
    }

    pub fn add_broadcast_func(
        &mut self,
        func: Arc<dyn BroadcastFunc<S, S, S, S::Inner, S::Inner, S::Inner>>,
    ) -> String {
        let fstr = func.as_str();
        self.broadcast_funcs.insert(fstr.clone(), func);
        fstr
    }

    pub fn add_corresponding_dims(&mut self, dims: Vec<(i32, i32)>) -> usize {
        let unique_id = &dims as *const _ as usize;
        self.corresponding_dims.insert(unique_id, dims);
        unique_id
    }

    pub fn from_lazy_tensor_compositional(tensor: LazyTensor<S>) -> Self {
        let mut context = CpuBackendContext::default();
        let eval_id = context.build_expression(&tensor);
        context.eval_id = Some(eval_id);
        context
    }

    fn build_expression(&mut self, tensor: &LazyTensor<S>) -> Id {
        match tensor {
            LazyTensor::Tensor(t) => {
                let tensor_id = self.add_tensor(t.clone());
                self.egraph.add(CpuBackendLanguage::Tensor(tensor_id))
            }

            LazyTensor::Map { input, func } => {
                let input_id = self.build_expression(input);
                let func_name = self.add_map_func(func.clone());
                let func_id = self.egraph.add(CpuBackendLanguage::MapFunc(func_name));
                self.egraph
                    .add(CpuBackendLanguage::Map([input_id, func_id]))
            }

            LazyTensor::Reduce { input, dim, func } => {
                let input_id = self.build_expression(input);
                let func_name = self.add_reduce_func(func.clone());
                let func_id = self.egraph.add(CpuBackendLanguage::ReduceFunc(func_name));
                let dim_id = self.egraph.add(CpuBackendLanguage::Dim(*dim));
                self.egraph
                    .add(CpuBackendLanguage::Reduce([input_id, dim_id, func_id]))
            }

            LazyTensor::Broadcast {
                lhs_input,
                rhs_input,
                corresponding_dimensions,
                func,
            } => {
                let lhs_id = self.build_expression(lhs_input);
                let rhs_id = self.build_expression(rhs_input);
                let func_name = self.add_broadcast_func(func.clone());
                let func_id = self
                    .egraph
                    .add(CpuBackendLanguage::BroadcastFunc(func_name));
                let dims_id = self.add_corresponding_dims(corresponding_dimensions.clone());
                let corrdim_id = self
                    .egraph
                    .add(CpuBackendLanguage::CorrespondingDims(dims_id));

                self.egraph.add(CpuBackendLanguage::Broadcast([
                    lhs_id, rhs_id, corrdim_id, func_id,
                ]))
            }
        }
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
            rewrite!("fused-matmul";
                "(reduce (broadcast ?x ?y ?corrdims ?multiply_func) ?dim ?sum_func)" =>
                "(fused_matmul ?x ?y ?corrdims ?dim)"
                if is_matmul("?multiply_func".parse().unwrap(), "?sum_func".parse().unwrap())
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

            CpuBackendLanguage::Broadcast([lhs_id, rhs_id, corrdims_id, func_id]) => {
                let lhs_input = self.execute_node(&expr[*lhs_id], expr);
                let rhs_input = self.execute_node(&expr[*rhs_id], expr);

                let corrdims_lookup = match &expr[*corrdims_id] {
                    CpuBackendLanguage::CorrespondingDims(d) => *d,
                    _ => panic!("Expected corresponding dimensions"),
                };
                let corrdims = self
                    .corresponding_dims
                    .get(&corrdims_lookup)
                    .expect("Corresponding dimensions not found in context");

                let func_lookup = match &expr[*func_id] {
                    CpuBackendLanguage::BroadcastFunc(f) => f,
                    v => panic!("Expected broadcast func {:?}", v),
                };
                let func = self
                    .broadcast_funcs
                    .get(func_lookup)
                    .expect("Broadcast function not found in context");

                Arc::new(Tensor::new(
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

            CpuBackendLanguage::FusedMatmul([lhs_id, rhs_id, corrdims_id, dim_id]) => {
                println!("{}", lhs_id);
                println!("{}", rhs_id);
                let lhs_input = self.execute_node(&expr[*lhs_id], expr);
                let rhs_input = self.execute_node(&expr[*rhs_id], expr);

                let corrdims_lookup = match &expr[*corrdims_id] {
                    CpuBackendLanguage::CorrespondingDims(d) => *d,
                    _ => panic!("Expected corresponding dimensions"),
                };

                let corrdims = self
                    .corresponding_dims
                    .get(&corrdims_lookup)
                    .expect("Corresponding dimensions not found in context");

                let dim = match &expr[*dim_id] {
                    CpuBackendLanguage::Dim(d) => *d,
                    _ => panic!("Expected dimension"),
                };

                self.execute_fused_matmul(&lhs_input, &rhs_input, corrdims, dim)
            }

            CpuBackendLanguage::Output(input_id) => self.execute_node(&expr[*input_id], expr),

            _ => panic!("Cannot execute leaf node directly"),
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

    fn execute_fused_matmul(
        &self,
        lhs: &Arc<Tensor<S>>,
        rhs: &Arc<Tensor<S>>,
        corrdims: &[(i32, i32)],
        dim: i32,
    ) -> Arc<Tensor<S>> {
        // For matmul, we assume the multiply and sum functions are standard
        let multiply_func = self
            .broadcast_funcs
            .get(&format!(
                "CpuMultiply({}, {} -> {})",
                std::any::type_name::<S>(),
                std::any::type_name::<S>(),
                std::any::type_name::<S>()
            ))
            .expect("Multiply function not found in context");
        let sum_func = self
            .reduce_funcs
            .get(&format!(
                "CpuSum({} -> {})",
                std::any::type_name::<S>(),
                std::any::type_name::<S>()
            ))
            .expect("Sum function not found in context");

        // Perform the broadcast (element-wise multiplication)
        let broadcasted = multiply_func.call(
            &lhs.layout,
            &lhs.storage,
            &rhs.layout,
            &rhs.storage,
            corrdims,
        );

        let broadcasted_layout = lhs.layout.broadcast(
            &rhs.layout,
            &lhs.layout
                .signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
                    &rhs.layout,
                    corrdims,
                ),
        );

        let reduced_storage = sum_func.call(&broadcasted_layout, &broadcasted, dim);

        Arc::new(Tensor::new(
            broadcasted_layout.reduce(broadcasted_layout.signed_dim_to_unsigned_dim(dim)),
            reduced_storage,
        ))
    }
}

impl<S: Storage> Default for CpuBackendContext<S> {
    fn default() -> Self {
        Self {
            tensors: HashMap::new(),
            map_funcs: HashMap::new(),
            reduce_funcs: HashMap::new(),
            broadcast_funcs: HashMap::new(),
            corresponding_dims: HashMap::new(),
            egraph: egg::EGraph::default(),
            eval_id: None,
            rewrites: Vec::new(),
        }
    }
}

impl<S: Storage> From<LazyTensor<S>> for CpuBackendContext<S> {
    fn from(tensor: LazyTensor<S>) -> Self {
        CpuBackendContext::from_lazy_tensor_compositional(tensor)
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
            CpuBackendLanguage::Broadcast(args) => costs(args[0]) + costs(args[1]) + 25,
            CpuBackendLanguage::FusedMatmul(args) => costs(args[0]) + costs(args[1]) + 30,
            CpuBackendLanguage::CorrespondingDims(_) => 0,
            CpuBackendLanguage::BroadcastFunc(_) => 0,
        }
    }
}
