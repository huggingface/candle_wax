use core::tensor::Tensor;
use std::sync::Arc;

use crate::context::BackendContext;

pub trait EggNodeExecutor<B: BackendContext> {
    fn execute_node(
        &self,
        node: &B::BackendLanguage,
        expr: &egg::RecExpr<B::BackendLanguage>,
        context: &B,
    ) -> Result<Arc<Tensor<B::BackendStorage>>, B::BackendError>;
}
