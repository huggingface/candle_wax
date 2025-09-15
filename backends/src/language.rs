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
        "tensor" = Tensor([Id; 2]), // [tensor_id, shape_id]
        "map" = Map([Id; 2]),        // [input_expr, func_id]
        "reduce" = Reduce([Id; 3]),  // [input_expr, func_id, dim]
        "broadcast" = Broadcast([Id; 4]), // [lhs_input_expr, rhs_input_expr, func_id, corresponding_dims]
        "output" = Output(Id), // input_expr

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
        // Shapes should look like [1, 2, 3, 4] etc
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

// #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
// pub enum ShapeDim {
//     Elided,
//     Unknown,
//     Known(usize),
//     Symbol(String),
// }

// #[derive(Debug, Eq, PartialOrd, Ord, Hash, Clone)]
// pub struct Shape {
//     pub shape: Vec<ShapeDim>,
// }

// pub enum ElidedStatus {
//     Leading,
//     Trailing,
//     None,
// }

// impl Shape {
//     pub fn is_elided(&self) -> ElidedStatus {
//         let leading = matches!(self.shape.first(), Some(ShapeDim::Elided));
//         let trailing = matches!(self.shape.last(), Some(ShapeDim::Elided));
//         match (leading, trailing) {
//             (true, false) => ElidedStatus::Leading,
//             (false, true) => ElidedStatus::Trailing,
//             (false, false) => ElidedStatus::None,
//             _ => unreachable!("Shape cannot have elided dimensions at both start and end"),
//         }
//     }
// }

// impl Display for Shape {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "Shape({:?})", self.shape)
//     }
// }

// impl FromStr for Shape {
//     type Err = ();

//     fn from_str(s: &str) -> Result<Self, Self::Err> {
//         // Examples
//         // [1 2 3 4]
//         // [1 ? 2 ?]
//         // [... 4]
//         // [1 2 ...]
//         // [1 ? ... ]
//         // elipsis can come at beginning or end of shape
//         // ? is an unknown dimension
//         let s = s.trim();

//         // Must start with '[' and end with ']'
//         if !s.starts_with('[') || !s.ends_with(']') {
//             return Err(());
//         }

//         // Extract content between brackets
//         let content = &s[1..s.len() - 1].trim();

//         // Handle empty brackets
//         if content.is_empty() {
//             return Ok(Shape { shape: vec![] });
//         }

//         let mut shape_dims = Vec::new();
//         let mut tokens: Vec<&str> = content.split_whitespace().collect();

//         // Check for leading ellipsis
//         let has_leading_ellipsis = !tokens.is_empty() && tokens[0] == "...";
//         if has_leading_ellipsis {
//             shape_dims.push(ShapeDim::Elided);
//             tokens.remove(0);
//         }

//         // Check for trailing ellipsis (need to check before processing other tokens)
//         let has_trailing_ellipsis = !tokens.is_empty() && tokens[tokens.len() - 1] == "...";
//         if has_trailing_ellipsis && has_leading_ellipsis {
//             return Err(()); // Ellipsis cannot be both at start and end
//         }

//         if has_trailing_ellipsis {
//             tokens.pop();
//         }

//         // Process remaining tokens
//         for token in tokens {
//             match token {
//                 "?" => shape_dims.push(ShapeDim::Unknown),
//                 "..." => return Err(()), // Ellipsis only allowed at start or end
//                 num_str => match num_str.parse::<usize>() {
//                     Ok(num) => shape_dims.push(ShapeDim::Known(num)),
//                     Err(_) => shape_dims.push(ShapeDim::Symbol(num_str.to_string())),
//                 },
//             }
//         }

//         // Add trailing ellipsis if present
//         if has_trailing_ellipsis {
//             shape_dims.push(ShapeDim::Elided);
//         }

//         Ok(Shape { shape: shape_dims })
//     }
// }

// impl PartialEq for Shape {
//     fn eq(&self, other: &Self) -> bool {
//         match (self.is_elided(), other.is_elided()) {
//             (ElidedStatus::None, ElidedStatus::None) => {
//                 if self.shape.len() != other.shape.len() {
//                     return false;
//                 }
//                 for (dim1, dim2) in self.shape.iter().zip(other.shape.iter()) {
//                     match (dim1, dim2) {
//                         (ShapeDim::Known(n1), ShapeDim::Known(n2)) => {
//                             if n1 != n2 {
//                                 return false;
//                             }
//                         }
//                         (ShapeDim::Symbol(s1), ShapeDim::Symbol(s2)) => {
//                             if s1 != s2 {
//                                 return false;
//                             }
//                         }

//                         | (ShapeDim::Known(_), ShapeDim::Unknown)
//                         | (ShapeDim::Known(_), ShapeDim::Symbol(_))
//                         | (ShapeDim::Unknown, ShapeDim::Unknown)
//                         | (ShapeDim::Unknown, ShapeDim::Known(_))
//                         | (ShapeDim::Unknown, ShapeDim::Symbol(_))
//                         | (ShapeDim::Symbol(_), ShapeDim::Unknown)
//                         | (ShapeDim::Symbol(_), ShapeDim::Known(_)) => {}

//                         _ => return false, // Elided dimensions should not appear here
//                     }
//                 }
//                 true
//             }
//             (ElidedStatus::Trailing, ElidedStatus::Trailing)
//             | (ElidedStatus::Trailing, ElidedStatus::None)
//             | (ElidedStatus::None, ElidedStatus::Trailing) => {
//                 for (dim1, dim2) in self.shape.iter().zip(other.shape.iter()) {
//                     match (dim1, dim2) {
//                         (ShapeDim::Known(n1), ShapeDim::Known(n2)) => {
//                             if n1 != n2 {
//                                 return false;
//                             }
//                         }
//                         (ShapeDim::Symbol(s1), ShapeDim::Symbol(s2)) => {
//                             if s1 != s2 {
//                                 return false;
//                             }
//                         }

//                         | (ShapeDim::Known(_), ShapeDim::Unknown)
//                         | (ShapeDim::Known(_), ShapeDim::Symbol(_))
//                         | (ShapeDim::Unknown, ShapeDim::Unknown)
//                         | (ShapeDim::Unknown, ShapeDim::Known(_))
//                         | (ShapeDim::Unknown, ShapeDim::Symbol(_))
//                         | (ShapeDim::Symbol(_), ShapeDim::Unknown)
//                         | (ShapeDim::Symbol(_), ShapeDim::Known(_)) => {}

//                         (ShapeDim::Elided, _) | (_, ShapeDim::Elided) => break, // Stop comparison at elided
//                     }
//                 }
//                 true
//             }
//             (ElidedStatus::Leading, ElidedStatus::Leading)
//             | (ElidedStatus::Leading, ElidedStatus::None)
//             | (ElidedStatus::None, ElidedStatus::Leading) => {
//                 for (dim1, dim2) in self.shape.iter().rev().zip(other.shape.iter().rev()) {
//                     match (dim1, dim2) {
//                         (ShapeDim::Known(n1), ShapeDim::Known(n2)) => {
//                             if n1 != n2 {
//                                 return false;
//                             }
//                         }
//                         (ShapeDim::Symbol(s1), ShapeDim::Symbol(s2)) => {
//                             if s1 != s2 {
//                                 return false;
//                             }
//                         }

//                         | (ShapeDim::Known(_), ShapeDim::Unknown)
//                         | (ShapeDim::Known(_), ShapeDim::Symbol(_))
//                         | (ShapeDim::Unknown, ShapeDim::Unknown)
//                         | (ShapeDim::Unknown, ShapeDim::Known(_))
//                         | (ShapeDim::Unknown, ShapeDim::Symbol(_))
//                         | (ShapeDim::Symbol(_), ShapeDim::Unknown)
//                         | (ShapeDim::Symbol(_), ShapeDim::Known(_)) => {}

//                         (ShapeDim::Elided, _) | (_, ShapeDim::Elided) => break, // Stop comparison at elided
//                     }
//                 }
//                 true
//             }
//             (ElidedStatus::Leading, ElidedStatus::Trailing)
//             | (ElidedStatus::Trailing, ElidedStatus::Leading) => true,
//         }
//     }
// }

// impl From<&Layout> for Shape {
//     fn from(layout: &Layout) -> Self {
//         Shape {
//             shape: layout.shape.iter().map(|&d| ShapeDim::Known(d)).collect(),
//         }
//     }
// }
