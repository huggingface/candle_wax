use egg::{CostFunction, Id};

use super::language::CpuBackendLanguage;

#[derive(Default)]
pub struct CpuBackendCost;

impl CostFunction<CpuBackendLanguage> for CpuBackendCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &CpuBackendLanguage, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            // Leaf nodes have no cost
            CpuBackendLanguage::MapFunc(_) => 0,
            CpuBackendLanguage::ReduceFunc(_) => 0,
            CpuBackendLanguage::BroadcastFunc(_) => 0,
            CpuBackendLanguage::Tensor(_) => 0,
            CpuBackendLanguage::Dim(_) => 0,
            CpuBackendLanguage::CorrespondingDims(_) => 0,

            // Basic operations
            CpuBackendLanguage::Map(args) => costs(args[0]) + 10,
            CpuBackendLanguage::Reduce(args) => costs(args[0]) + 15,
            CpuBackendLanguage::Broadcast(args) => costs(args[0]) + costs(args[1]) + 25,

            // Fused operations (more efficient)
            CpuBackendLanguage::FusedMapReduce(args) => costs(args[0]) + 20,
            CpuBackendLanguage::FusedReduceMap(args) => costs(args[0]) + 20,
            CpuBackendLanguage::FusedMatmul(args) => costs(args[0]) + costs(args[1]) + 30,

            // Output wrapper
            CpuBackendLanguage::Output(id) => costs(*id),
        }
    }
}
