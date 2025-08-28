use egg::{
    CostFunction, EGraph, Extractor, Id, Language, Rewrite, Runner, Subst, define_language, rewrite,
};
use std::{collections::HashMap, sync::Arc, usize};

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

fn is_matmul(
    f1: &'static str,
    f2: &'static str,
) -> impl Fn(&mut EGraph<CpuBackendLanguage, ()>, Id, &Subst) -> bool {
    let var1 = f1.parse().unwrap();
    let var2 = f2.parse().unwrap();

    let mul = CpuBackendLanguage::BroadcastFunc("Multiply".to_string());
    let sum = CpuBackendLanguage::ReduceFunc("Sum".to_string());

    move |egraph, _, subst| {
        egraph[subst[var1]].nodes.contains(&mul) && egraph[subst[var2]].nodes.contains(&sum)
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

    pub fn combine(&mut self, other: CpuBackendContext<S>) -> HashMap<Id, Id> {
        // Merge all the hash maps
        self.tensors.extend(other.tensors);
        self.map_funcs.extend(other.map_funcs);
        self.reduce_funcs.extend(other.reduce_funcs);
        self.broadcast_funcs.extend(other.broadcast_funcs);
        self.corresponding_dims.extend(other.corresponding_dims);

        // For e-graph merging, we need to rebuild the other e-graph's expressions
        // in our e-graph context. This is more complex than simple ID mapping.

        // Create a mapping from other's class IDs to our class IDs
        let mut id_mapping = HashMap::new();

        // We need to process nodes in topological order to ensure dependencies are met
        // For simplicity, we'll use a worklist approach
        let mut worklist: Vec<Id> = other.egraph.classes().map(|class| class.id).collect();
        let mut processed = std::collections::HashSet::new();

        while !worklist.is_empty() {
            let mut progress = false;
            let mut remaining = Vec::new();

            for &class_id in &worklist {
                if processed.contains(&class_id) {
                    continue;
                }

                let class = &other.egraph[class_id];
                let mut can_process = true;

                // Check if all children of all nodes in this class have been processed
                for node in &class.nodes {
                    for &child_id in node.children() {
                        if !processed.contains(&child_id) && !id_mapping.contains_key(&child_id) {
                            can_process = false;
                            break;
                        }
                    }
                    if !can_process {
                        break;
                    }
                }

                if can_process {
                    // Process this class
                    let representative_node = &class.nodes[0]; // Take first node as representative

                    // Map children to our ID space
                    let mapped_node = representative_node.clone().map_children(|child_id| {
                        id_mapping.get(&child_id).copied().unwrap_or(child_id)
                    });

                    // Add to our e-graph
                    let new_id = self.egraph.add(mapped_node);
                    id_mapping.insert(class_id, new_id);
                    processed.insert(class_id);
                    progress = true;

                    // Add all other nodes in the class to maintain equivalences
                    for node in &class.nodes[1..] {
                        let mapped_node = node.clone().map_children(|child_id| {
                            id_mapping.get(&child_id).copied().unwrap_or(child_id)
                        });
                        let another_id = self.egraph.add(mapped_node);
                        self.egraph.union(new_id, another_id);
                    }
                } else {
                    remaining.push(class_id);
                }
            }

            if !progress && !remaining.is_empty() {
                // If we can't make progress, there might be cycles or missing dependencies
                // For now, we'll break to avoid infinite loops
                break;
            }

            worklist = remaining;
        }

        // Update eval_id if the other context had one
        if let Some(other_eval_id) = other.eval_id {
            if let Some(&mapped_eval_id) = id_mapping.get(&other_eval_id) {
                match self.eval_id {
                    Some(current_eval_id) => {
                        // If we have both eval_ids, we might want to union them or handle differently
                        // For now, we'll keep the current one and ignore the other
                        // Alternatively: self.egraph.union(current_eval_id, mapped_eval_id);
                    }
                    None => {
                        self.eval_id = Some(mapped_eval_id);
                    }
                }
            }
        }

        id_mapping
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
                if is_matmul("?multiply_func", "?sum_func")
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

            CpuBackendLanguage::Dim(_) => {
                panic!("Cannot execute dimension node directly")
            }

            CpuBackendLanguage::MapFunc(_) => {
                panic!("Cannot execute map func node directly")
            }

            CpuBackendLanguage::ReduceFunc(_) => {
                panic!("Cannot execute reduce func node directly")
            }

            CpuBackendLanguage::BroadcastFunc(_) => {
                panic!("Cannot execute broadcast func node directly")
            }

            CpuBackendLanguage::CorrespondingDims(_) => {
                panic!("Cannot execute corresponding dimensions node directly")
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

    fn execute_fused_matmul(
        &self,
        lhs: &Arc<Tensor<S>>,
        rhs: &Arc<Tensor<S>>,
        corrdims: &Vec<(i32, i32)>,
        dim: i32,
    ) -> Arc<Tensor<S>> {
        // For matmul, we assume the multiply and sum functions are standard
        let multiply_func = self
            .broadcast_funcs
            .get("Multiply")
            .expect("Multiply function not found in context");
        let sum_func = self
            .reduce_funcs
            .get("Sum")
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
            LazyTensor::Broadcast {
                lhs_input,
                rhs_input,
                corresponding_dimensions,
                func,
            } => {
                let mut context = CpuBackendContext::from(*lhs_input);
                let rhs_context = CpuBackendContext::from(*rhs_input);
                let rhs_id = rhs_context.eval_id.expect("RHS eval_id missing");
                let mapping = context.combine(rhs_context);
                let rhs_id = *mapping.get(&rhs_id).expect("RHS ID not found in mapping");
                let func_id = context.add_broadcast_func(func);
                let reduce_func_id = context
                    .egraph
                    .add(CpuBackendLanguage::BroadcastFunc(func_id));
                let corrdims_id = context.add_corresponding_dims(corresponding_dimensions);
                let corrdim_ids = context
                    .egraph
                    .add(CpuBackendLanguage::CorrespondingDims(corrdims_id));
                let id: Id = context.egraph.add(CpuBackendLanguage::Broadcast([
                    context.eval_id.unwrap(),
                    rhs_id, // Note: This should be the eval_id of rhs_input after combining
                    corrdim_ids,
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
            CpuBackendLanguage::Broadcast(args) => costs(args[0]) + costs(args[1]) + 25,
            CpuBackendLanguage::FusedMatmul(args) => costs(args[0]) + costs(args[1]) + 30,
            CpuBackendLanguage::CorrespondingDims(_) => 0,
            CpuBackendLanguage::BroadcastFunc(_) => 0,
        }
    }
}
