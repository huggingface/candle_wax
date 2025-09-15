use std::fmt::Debug;
use std::hash::Hash;

use egg::{EGraph, Id, Rewrite, Subst, Var, define_language, rewrite};
use regex::Regex;

use crate::language::{CoreLanguage, CorrespondingDims, FunctionLookup, TensorRef};

define_language! {
    pub enum CpuBackendLanguage {
        "map" = Map([Id; 2]),        // [input_expr, func_id]
        "reduce" = Reduce([Id; 3]),  // [input_expr, func_id, dim]
        "broadcast" = Broadcast([Id; 4]), // [lhs_input_expr, rhs_input_expr, func_id, corresponding_dims]

        "fused_batched_matmul" = FusedMatmul([Id; 2]),        // [lhs_input_expr, rhs_input_expr]
        "fused_softmax" = FusedSoftmax([Id; 1]),      // [input_expr]

        "output" = Output(Id), // input_expr

        MapFunc(FunctionLookup),
        ReduceFunc(FunctionLookup),
        BroadcastFunc(FunctionLookup),

        Tensor(TensorRef),
        CorrespondingDims(CorrespondingDims),
        Dim(i32),
    }
}

impl From<CoreLanguage> for CpuBackendLanguage {
    fn from(core: CoreLanguage) -> Self {
        match core {
            CoreLanguage::Map(args) => CpuBackendLanguage::Map(args),
            CoreLanguage::Reduce(args) => CpuBackendLanguage::Reduce(args),
            CoreLanguage::Broadcast(args) => CpuBackendLanguage::Broadcast(args),
            CoreLanguage::Output(args) => CpuBackendLanguage::Output(args),
            CoreLanguage::MapFunc(func) => CpuBackendLanguage::MapFunc(func),
            CoreLanguage::ReduceFunc(func) => CpuBackendLanguage::ReduceFunc(func),
            CoreLanguage::BroadcastFunc(func) => CpuBackendLanguage::BroadcastFunc(func),
            CoreLanguage::Tensor(tensor) => CpuBackendLanguage::Tensor(tensor),
            CoreLanguage::CorrespondingDims(dims) => CpuBackendLanguage::CorrespondingDims(dims),
            CoreLanguage::Dim(dim) => CpuBackendLanguage::Dim(dim),
        }
    }
}

pub fn rewrites() -> Vec<Rewrite<CpuBackendLanguage, ()>> {
    vec![
        rewrite!("fused-batched-matmul";
            "(reduce (broadcast ?x ?y  ?multiply_func ?corrdims) ?sum_func ?dim)" =>
            "(fused_batched_matmul ?x ?y)"
            if is_matmul("?x".parse().unwrap(), "?y".parse().unwrap(), "?multiply_func".parse().unwrap(), "?sum_func".parse().unwrap(), "?corrdims".parse().unwrap(), "?dim".parse().unwrap())
        ),
        rewrite!("fused-softmax";
            "(broadcast (map ?x ?exp_func) (reduce (map ?x ?exp_func) ?sum_func ?dim) ?div_func ?corrdims)" =>
            "(fused_softmax ?x)"
            if is_softmax("?x".parse().unwrap(), "?exp_func".parse().unwrap(), "?sum_func".parse().unwrap(), "?div_func".parse().unwrap(), "?dim".parse().unwrap(), "?corrdims".parse().unwrap())
        ),
    ]
}

fn is_matmul(
    tensor1: Var,
    tensor2: Var,
    f1: Var,
    f2: Var,
    corrdims: Var,
    dim: Var,
) -> impl Fn(&mut EGraph<CpuBackendLanguage, ()>, Id, &Subst) -> bool {
    let mul_pattern = Regex::new(r".*Multiply.*").unwrap();
    let sum_pattern = Regex::new(r".*Sum.*").unwrap();

    move |egraph, _, subst| {
        let f1_matches = egraph[subst[f1]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::BroadcastFunc(func_lookup) = node {
                mul_pattern.is_match(func_lookup.func_type.as_str())
            } else {
                false
            }
        });

        let f2_matches = egraph[subst[f2]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::ReduceFunc(func_lookup) = node {
                sum_pattern.is_match(func_lookup.func_type.as_str())
            } else {
                false
            }
        });

        // let tensor_a_shape = egraph[subst[tensor1]]
        //     .nodes
        //     .iter()
        //     .find_map(|node| {
        //         if let CpuBackendLanguage::Tensor(tensor_ref) = node {
        //             Some(tensor_ref.shape.clone())
        //         } else {
        //             None
        //         }
        //     })
        //     .unwrap();

        // let tensor_b_shape = egraph[subst[tensor2]]
        //     .nodes
        //     .iter()
        //     .find_map(|node| {
        //         if let CpuBackendLanguage::Tensor(tensor_ref) = node {
        //             Some(tensor_ref.shape.clone())
        //         } else {
        //             None
        //         }
        //     })
        //     .unwrap();

        // let cordims_matches = egraph[subst[corrdims]].nodes.iter().any(|node| {
        //     if let CpuBackendLanguage::CorrespondingDims(dims) = node {
        //         dims.0.len() == 1
        //             && (dims.0[0] == (-2, -1)
        //                 || (dims.0[0].0 as usize == (tensor_a_shape.len() - 1)
        //                     && dims.0[0].1 as usize == (tensor_b_shape.len() - 2)))
        //     } else {
        //         false
        //     }
        // });

        // let dims_matches = egraph[subst[dim]].nodes.iter().any(|node| {
        //     if let CpuBackendLanguage::Dim(d) = node {
        //         *d == -2 || (*d as usize) == (tensor_a_shape.len() + tensor_b_shape.len() - 3)
        //     } else {
        //         false
        //     }
        // });

        f1_matches && f2_matches // && cordims_matches && dims_matches
    }
}

fn is_softmax(
    tensor: Var,
    f1: Var,
    f2: Var,
    f3: Var,
    dim: Var,
    corrdims: Var,
) -> impl Fn(&mut EGraph<CpuBackendLanguage, ()>, Id, &Subst) -> bool {
    let exp_pattern = Regex::new(r".*Exp.*").unwrap();
    let sum_pattern = Regex::new(r".*Sum.*").unwrap();
    let div_pattern = Regex::new(r".*Divide.*").unwrap();

    move |egraph, _, subst| {
        let f1_matches = egraph[subst[f1]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::MapFunc(func_lookup) = node {
                exp_pattern.is_match(func_lookup.func_type.as_str())
            } else {
                false
            }
        });

        let f2_matches = egraph[subst[f2]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::ReduceFunc(func_lookup) = node {
                sum_pattern.is_match(func_lookup.func_type.as_str())
            } else {
                false
            }
        });

        let f3_matches = egraph[subst[f3]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::BroadcastFunc(func_lookup) = node {
                div_pattern.is_match(func_lookup.func_type.as_str())
            } else {
                false
            }
        });

        // let tensor_shape = egraph[subst[tensor]]
        //     .nodes
        //     .iter()
        //     .find_map(|node| {
        //         if let CpuBackendLanguage::Tensor(tensor_ref) = node {
        //             Some(tensor_ref.shape.clone())
        //         } else {
        //             None
        //         }
        //     })
        //     .unwrap();

        // let corrdims_matches = egraph[subst[corrdims]].nodes.iter().any(|node| {
        //     if let CpuBackendLanguage::CorrespondingDims(dims) = node {
        //         dims.0.len() == 1
        //             && (dims.0[0] == (-2, -1)
        //                 || (dims.0[0].0 as usize == (tensor_shape.len() - 1)
        //                     && dims.0[0].1 as usize == (tensor_shape.len() - 2)))
        //     } else {
        //         false
        //     }
        // });

        // let dims_matches = egraph[subst[dim]].nodes.iter().any(|node| {
        //     if let CpuBackendLanguage::Dim(d) = node {
        //         *d == -1 || (*d as usize) == (tensor_shape.len() - 1)
        //     } else {
        //         false
        //     }
        // });

        f1_matches && f2_matches && f3_matches //&& corrdims_matches && dims_matches
    }
}
