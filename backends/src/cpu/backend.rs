use core::{
    Layout,
    backends::{
        Backend, LazyBackend,
        broadcast::{Broadcast, BroadcastFunc, BroadcastFuncSame},
        map::{Map, MapFunc},
        reduce::{Reduce, ReduceFunc},
    },
    storage::Storage,
    tensor::{LazyTensor, Tensor},
};
use egg::Id;
use macros::BackendOps;
use std::sync::Arc;

use crate::context::BackendContext;

use super::{
    context::CpuBackendContext,
    cost::CpuBackendCost,
    language::{CpuBackendLanguage, rewrites},
};

#[allow(dead_code)]
#[derive(Debug)]
pub enum CpuBackendError {
    TensorNotFound(usize),
    FunctionNotFound(String),
    UnexpectedNodeType { expected: String, found: String },
    CorrespondingDimensionsNotFound(usize),
    NoEvaluationId,
    EmptyExpression,
    ExecutionFailed(String),
}

#[derive(BackendOps)]
pub struct CpuBackend {}

impl Backend for CpuBackend {}

impl LazyBackend for CpuBackend {
    fn eval<S: Storage>(tensor: LazyTensor<S>) -> Tensor<S> {
        let mut context = CpuBackendContext::from(tensor);

        context.eval_node_id = Some(
            context
                .egraph
                .add(CpuBackendLanguage::Output(context.eval_node_id.unwrap())),
        );

        context.add_rewrites(&rewrites());
        context.optimize();

        match context.evaluate(CpuBackendCost) {
            Ok(result) => result.as_ref().clone(),
            Err(e) => panic!("CPU Backend evaluation failed: {:?}", e),
        }
    }
}

pub struct CpuExpressionBuilder<S: Storage> {
    context: CpuBackendContext<S>,
}

impl<S: Storage> CpuExpressionBuilder<S> {
    pub fn new() -> Self {
        Self {
            context: CpuBackendContext::default(),
        }
    }

    pub fn build_from_lazy_tensor(mut self, tensor: &LazyTensor<S>) -> CpuBackendContext<S> {
        let eval_node_id = self.build_expression(tensor);
        self.context.set_eval_node_id(eval_node_id);
        self.context
    }

    fn build_expression(&mut self, tensor: &LazyTensor<S>) -> Id {
        match tensor {
            LazyTensor::Tensor(t) => self.build_tensor_expr(t.clone()),
            LazyTensor::Map { input, func, .. } => self.build_map_expr(input, func.clone()),
            LazyTensor::Reduce {
                input, dim, func, ..
            } => self.build_reduce_expr(input, *dim, func.clone()),
            LazyTensor::Broadcast {
                lhs_input,
                rhs_input,
                corresponding_dimensions,
                func,
                ..
            } => self.build_broadcast_expr(
                lhs_input,
                rhs_input,
                corresponding_dimensions.clone(),
                func.clone(),
            ),
        }
    }

    fn build_tensor_expr(&mut self, tensor: Arc<Tensor<S>>) -> Id {
        self.context.add_tensor(tensor)
    }

    fn build_map_expr(
        &mut self,
        input: &LazyTensor<S>,
        func: Arc<dyn MapFunc<S, S, <S as Storage>::Inner, <S as Storage>::Inner>>,
    ) -> Id {
        let input_id = self.build_expression(input);
        self.context.add_map(input_id, func)
    }

    fn build_reduce_expr(
        &mut self,
        input: &LazyTensor<S>,
        dim: i32,
        func: Arc<dyn ReduceFunc<S, S, <S as Storage>::Inner, <S as Storage>::Inner>>,
    ) -> Id {
        let input_id = self.build_expression(input);
        self.context.add_reduce(input_id, func, dim)
    }

    fn build_broadcast_expr(
        &mut self,
        lhs_input: &LazyTensor<S>,
        rhs_input: &LazyTensor<S>,
        corresponding_dimensions: Vec<(i32, i32)>,
        func: Arc<BroadcastFuncSame<S>>,
    ) -> Id {
        let lhs_id = self.build_expression(lhs_input);
        let rhs_id = self.build_expression(rhs_input);
        self.context
            .add_broadcast(lhs_id, rhs_id, func, corresponding_dimensions)
    }
}

impl<S: Storage> From<LazyTensor<S>> for CpuBackendContext<S> {
    fn from(tensor: LazyTensor<S>) -> Self {
        CpuExpressionBuilder::new().build_from_lazy_tensor(&tensor)
    }
}
