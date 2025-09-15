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

use crate::context::{BackendContext, CoreContext};

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
    type LazyBackendError = CpuBackendError;

    fn eval<S: Storage>(tensor: LazyTensor<S>) -> Result<Tensor<S>, CpuBackendError> {
        let mut context = CpuBackendContext::from(tensor);

        context.eval_node_id = Some(
            context
                .egraph
                .add(CpuBackendLanguage::Output(context.eval_node_id.unwrap())),
        );

        context.add_rewrites(&rewrites());
        context.optimize();

        match context.evaluate(
            CpuBackendCost,
            context
                .eval_node_id
                .ok_or(CpuBackendError::NoEvaluationId)?,
        ) {
            Ok(result) => Ok(result.as_ref().clone()),
            Err(e) => Err(e),
        }
    }
}

impl<S: Storage> From<LazyTensor<S>> for CpuBackendContext<S> {
    fn from(tensor: LazyTensor<S>) -> Self {
        CoreContext::from(&tensor).into()
    }
}
