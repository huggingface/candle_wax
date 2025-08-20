pub mod map;
pub use map::{
    Map,
    MapFunc,
    relu::Relu
};

pub mod reduce;
pub use reduce::{
    Reduce,
    ReduceFunc,
    sum::Sum
};