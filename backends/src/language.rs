use core::Layout;
use egg::{Id, define_language};
use regex::Regex;
use std::hash::Hash;
use std::{
    fmt::{Debug, Display},
    str::FromStr,
};

define_language! {
    pub enum CoreLanguage {
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
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct CorrespondingDims(pub Vec<(i32, i32)>);

impl Display for CorrespondingDims {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CorrespondingDims({:?})", self.0)
    }
}

impl FromStr for CorrespondingDims {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Corresponding dims should look for a string that looks like:
        // [(0, 1), (-3, -2), ...] etc
        let re = Regex::new(
            r"\[\s*(\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*,\s*)*(\(\s*-?\d+\s*,\s*-?\d+\s*\))\s*\]",
        )
        .unwrap();
        let inner_re = Regex::new(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)").unwrap();
        if let Some(cap) = re.captures(s) {
            let mut dims = Vec::new();
            for i in 1..cap.len() {
                let dim = cap[i].to_string();
                if let Some(dim_cap) = inner_re.captures(&dim) {
                    let id1 = dim_cap[1].parse::<i32>().unwrap();
                    let id2 = dim_cap[2].parse::<i32>().unwrap();
                    dims.push((id1, id2));
                }
            }
            Ok(CorrespondingDims(dims))
        } else {
            Err(())
        }
    }
}

impl From<Vec<(i32, i32)>> for CorrespondingDims {
    fn from(value: Vec<(i32, i32)>) -> Self {
        CorrespondingDims(value)
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct FunctionLookup {
    pub id: usize,
    pub func_type: String,
}

impl Display for FunctionLookup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.func_type, self.id)
    }
}

impl FromStr for FunctionLookup {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = Regex::new(r"\s*([a-zA-Z]+[a-zA-Z0-9]*)\(\s*(\d+)\s*\)").unwrap();
        if let Some(cap) = re.captures(s) {
            let func_type = cap[1].to_string();
            let id = cap[2].parse::<usize>().unwrap();
            Ok(FunctionLookup { id, func_type })
        } else {
            Err(())
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct TensorRef {
    pub id: usize,
}

impl Display for TensorRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorRef({:?})", self.id)
    }
}

impl FromStr for TensorRef {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = Regex::new(r"TensorRef\(\s*(\d+)\s*\)").unwrap();
        if let Some(cap) = re.captures(s) {
            let id = cap[1].parse::<usize>().unwrap();
            Ok(TensorRef { id })
        } else {
            Err(())
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct Shape {
    pub shape: Vec<usize>,
}

impl From<&Layout> for Shape {
    fn from(layout: &Layout) -> Self {
        Shape {
            shape: layout.shape.clone(),
        }
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shape({:?})", self.shape)
    }
}

impl FromStr for Shape {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = Regex::new(r"\[\s*(\d+\s*,\s*)*(\d+)\s*\]").unwrap();
        let inner_re = Regex::new(r"\d+").unwrap();
        if let Some(cap) = re.captures(s) {
            let mut shape = Vec::new();
            for i in 1..cap.len() {
                let dim = cap[i].to_string();
                if let Some(dim_cap) = inner_re.captures(&dim) {
                    let id = dim_cap[0].parse::<usize>().unwrap();
                    shape.push(id);
                }
            }
            Ok(Shape { shape })
        } else {
            Err(())
        }
    }
}
