use std::fmt::Debug;
use std::hash::Hash;

use egg::{EGraph, Id, Rewrite, Subst, Var, define_language, rewrite};
use regex::Regex;

use crate::language::{CoreLanguage, CorrespondingDims, FunctionLookup, Shape, TensorRef};

define_language! {
    pub enum CpuBackendLanguage {
        "tensor" = Tensor([Id; 2]), // [tensor_id, out_shape_id]
        "map" = Map([Id; 3]),        // [input_expr, func_id, out_shape_id]
        "reduce" = Reduce([Id; 4]),  // [input_expr, func_id, dim, out_shape_id]
        "broadcast" = Broadcast([Id; 5]), // [lhs_input_expr, rhs_input_expr, func_id, corresponding_dims, out_shape_id]
        "output" = Output(Id), // [input_expr]

        MapFunc(FunctionLookup),
        ReduceFunc(FunctionLookup),
        BroadcastFunc(FunctionLookup),

        TensorRef(TensorRef),
        CorrespondingDims(CorrespondingDims),
        Dim(i32),
        Shape(Shape),

        "fused_batched_matmul" = FusedMatmul([Id; 2]),        // [lhs_input_expr, rhs_input_expr]
        "fused_softmax" = FusedSoftmax([Id; 1]),      // [input_expr]
    }
}

impl From<CoreLanguage> for CpuBackendLanguage {
    fn from(core: CoreLanguage) -> Self {
        match core {
            CoreLanguage::Tensor(args) => CpuBackendLanguage::Tensor(args),
            CoreLanguage::Map(args) => CpuBackendLanguage::Map(args),
            CoreLanguage::Reduce(args) => CpuBackendLanguage::Reduce(args),
            CoreLanguage::Broadcast(args) => CpuBackendLanguage::Broadcast(args),
            CoreLanguage::Output(args) => CpuBackendLanguage::Output(args),
            CoreLanguage::MapFunc(func) => CpuBackendLanguage::MapFunc(func),
            CoreLanguage::ReduceFunc(func) => CpuBackendLanguage::ReduceFunc(func),
            CoreLanguage::BroadcastFunc(func) => CpuBackendLanguage::BroadcastFunc(func),
            CoreLanguage::TensorRef(tensor_ref) => CpuBackendLanguage::TensorRef(tensor_ref),
            CoreLanguage::CorrespondingDims(dims) => CpuBackendLanguage::CorrespondingDims(dims),
            CoreLanguage::Dim(dim) => CpuBackendLanguage::Dim(dim),
            CoreLanguage::Shape(shape) => CpuBackendLanguage::Shape(shape),
        }
    }
}

pub fn rewrites() -> Vec<Rewrite<CpuBackendLanguage, ()>> {
    vec![
        rewrite!("fused-batched-matmul";
            "(reduce (broadcast ?x ?y ?multiply_func ?corrdims ?b_shape) ?sum_func ?dim ?r_shape)" =>
            "(fused_batched_matmul ?x ?y)"
            if is_matmul("?x".parse().unwrap(), "?y".parse().unwrap(), "?multiply_func".parse().unwrap(), "?sum_func".parse().unwrap(), "?corrdims".parse().unwrap(), "?dim".parse().unwrap())
        ),
        rewrite!("fused-softmax";
            "(broadcast (map ?x ?exp_func ?m_shape) (reduce (map ?x ?exp_func ?m_shape2) ?sum_func ?dim ?r_shape) ?div_func ?corrdims ?b_shape)" =>
            "(fused_softmax ?x)"
            if is_softmax("?x".parse().unwrap(), "?exp_func".parse().unwrap(), "?sum_func".parse().unwrap(), "?div_func".parse().unwrap(), "?dim".parse().unwrap(), "?corrdims".parse().unwrap())
        ),
    ]
}

fn is_matmul(
    input1: Var,
    input2: Var,
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
                mul_pattern.is_match(&func_lookup.hint)
            } else {
                false
            }
        });

        let f2_matches = egraph[subst[f2]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::ReduceFunc(func_lookup) = node {
                sum_pattern.is_match(&func_lookup.hint)
            } else {
                false
            }
        });

        let input_1_shape_id = egraph[subst[input1]]
            .nodes
            .iter()
            .find_map(|node| match node {
                CpuBackendLanguage::Tensor([_, shape_id]) => Some(shape_id),
                CpuBackendLanguage::Map([.., shape_id]) => Some(shape_id),
                CpuBackendLanguage::Reduce([.., shape_id]) => Some(shape_id),
                CpuBackendLanguage::Broadcast([.., shape_id]) => Some(shape_id),
                _ => None,
            })
            .unwrap();

        let input_1_shape = egraph[*input_1_shape_id]
            .nodes
            .iter()
            .find_map(|shape_node| {
                if let CpuBackendLanguage::Shape(shape) = shape_node {
                    Some(shape.shape.clone())
                } else {
                    None
                }
            })
            .unwrap();

        let input_2_shape_id = egraph[subst[input2]]
            .nodes
            .iter()
            .find_map(|node| match node {
                CpuBackendLanguage::Tensor([_, shape_id]) => Some(shape_id),
                CpuBackendLanguage::Map([.., shape_id]) => Some(shape_id),
                CpuBackendLanguage::Reduce([.., shape_id]) => Some(shape_id),
                CpuBackendLanguage::Broadcast([.., shape_id]) => Some(shape_id),
                _ => None,
            })
            .unwrap();

        let input_2_shape = egraph[*input_2_shape_id]
            .nodes
            .iter()
            .find_map(|node| {
                if let CpuBackendLanguage::Shape(shape) = node {
                    Some(shape.shape.clone())
                } else {
                    None
                }
            })
            .unwrap();

        let cordims_matches = egraph[subst[corrdims]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::CorrespondingDims(dims) = node {
                dims.0.len() == 1
                    && (dims.0[0] == (-2, -1)
                        || (dims.0[0].0 as usize == (input_1_shape.len() - 1)
                            && dims.0[0].1 as usize == (input_2_shape.len() - 2)))
            } else {
                false
            }
        });

        let dims_matches = egraph[subst[dim]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::Dim(d) = node {
                *d == -2 || (*d as usize) == (input_1_shape.len() + input_2_shape.len() - 3)
            } else {
                false
            }
        });

        f1_matches && f2_matches && cordims_matches && dims_matches
    }
}

fn is_softmax(
    input: Var,
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
                exp_pattern.is_match(&func_lookup.hint)
            } else {
                false
            }
        });

        let f2_matches = egraph[subst[f2]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::ReduceFunc(func_lookup) = node {
                sum_pattern.is_match(&func_lookup.hint)
            } else {
                false
            }
        });

        let f3_matches = egraph[subst[f3]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::BroadcastFunc(func_lookup) = node {
                div_pattern.is_match(&func_lookup.hint)
            } else {
                false
            }
        });

        let shape_id = egraph[subst[input]]
            .nodes
            .iter()
            .find_map(|node| match node {
                CpuBackendLanguage::Tensor([_, shape_id]) => Some(shape_id),
                CpuBackendLanguage::Map([.., shape_id]) => Some(shape_id),
                CpuBackendLanguage::Reduce([.., shape_id]) => Some(shape_id),
                CpuBackendLanguage::Broadcast([.., shape_id]) => Some(shape_id),
                _ => None,
            })
            .unwrap();

        let shape = egraph[*shape_id]
            .nodes
            .iter()
            .find_map(|node| {
                if let CpuBackendLanguage::Shape(shape) = node {
                    Some(shape.shape.clone())
                } else {
                    None
                }
            })
            .unwrap();

        let corrdims_matches = egraph[subst[corrdims]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::CorrespondingDims(dims) = node {
                dims.0.len() == 1
                    && (dims.0[0] == (-2, -1)
                        || (dims.0[0].0 as usize == (shape.len() - 1)
                            && dims.0[0].1 as usize == (shape.len() - 2)))
            } else {
                false
            }
        });

        let dims_matches = egraph[subst[dim]].nodes.iter().any(|node| {
            if let CpuBackendLanguage::Dim(d) = node {
                *d == -1 || (*d as usize) == (shape.len() - 1)
            } else {
                false
            }
        });

        f1_matches && f2_matches && f3_matches && corrdims_matches && dims_matches
    }
}
