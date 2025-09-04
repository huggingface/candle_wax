use egg::{EGraph, Id, Rewrite, Subst, Var, define_language, rewrite};
use regex::Regex;

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

pub fn rewrites() -> Vec<Rewrite<CpuBackendLanguage, ()>> {
    vec![
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
    ]
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
