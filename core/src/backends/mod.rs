pub(crate) mod op_traits;

pub(crate) mod cpu;

pub use cpu::backend::CpuBackend;
pub mod core_ops {
    pub use super::op_traits::{map::Map, reduce::Reduce};
}
pub mod map {
    pub use super::op_traits::map::{MapFunc, relu::Relu};
}
pub mod reduce {
    pub use super::op_traits::reduce::{ReduceFunc, sum::Sum};
}

pub trait Backend {}
