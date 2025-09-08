use core::{storage::Storage, tensor::Tensor};
use egg::Id;
use std::sync::Arc;

use crate::node_executor::EggNodeExecutor;

use super::{backend::CpuBackendError, context::CpuBackendContext, language::CpuBackendLanguage};

pub struct CpuExecutor;

impl<S: Storage> EggNodeExecutor<CpuBackendContext<S>> for CpuExecutor {
    fn execute_node(
        &self,
        node: &CpuBackendLanguage,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>, CpuBackendError> {
        match node {
            CpuBackendLanguage::Tensor(tensor_id) => context
                .tensors
                .get(tensor_id)
                .cloned()
                .ok_or(CpuBackendError::TensorNotFound(*tensor_id)),

            CpuBackendLanguage::Map([input_id, func_id]) => {
                self.execute_map(*input_id, *func_id, expr, context)
            }

            CpuBackendLanguage::Reduce([input_id, dim_id, func_id]) => {
                self.execute_reduce(*input_id, *dim_id, *func_id, expr, context)
            }

            CpuBackendLanguage::Broadcast([lhs_id, rhs_id, corrdims_id, func_id]) => {
                self.execute_broadcast(*lhs_id, *rhs_id, *corrdims_id, *func_id, expr, context)
            }

            CpuBackendLanguage::FusedMapReduce([input_id, map_func_id, dim_id, reduce_func_id]) => {
                self.execute_fused_map_reduce(
                    *input_id,
                    *map_func_id,
                    *dim_id,
                    *reduce_func_id,
                    expr,
                    context,
                )
            }

            CpuBackendLanguage::FusedReduceMap([input_id, dim_id, reduce_func_id, map_func_id]) => {
                self.execute_fused_reduce_map(
                    *input_id,
                    *dim_id,
                    *reduce_func_id,
                    *map_func_id,
                    expr,
                    context,
                )
            }

            CpuBackendLanguage::FusedMatmul([lhs_id, rhs_id, corrdims_id, dim_id]) => {
                self.execute_fused_matmul(*lhs_id, *rhs_id, *corrdims_id, *dim_id, expr, context)
            }

            CpuBackendLanguage::FusedSoftmax([input_id]) => {
                self.execute_fused_softmax(*input_id, expr, context)
            }

            CpuBackendLanguage::Output(input_id) => {
                self.execute_node(&expr[*input_id], expr, context)
            }

            _ => Err(CpuBackendError::UnexpectedNodeType {
                expected: "executable node".to_string(),
                found: format!("{:?}", node),
            }),
        }
    }
}

impl CpuExecutor {
    fn execute_map<S: Storage>(
        &self,
        input_id: Id,
        func_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>, CpuBackendError> {
        let input = self.execute_node(&expr[input_id], expr, context)?;

        let func_lookup = match &expr[func_id] {
            CpuBackendLanguage::MapFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "MapFunc".to_string(),
                    found: format!("{:?}", &expr[func_id]),
                });
            }
        };

        let func = context
            .map_funcs
            .get(func_lookup.as_str())
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(func_lookup.to_string()))?;

        Ok(Arc::new(Tensor::new(
            input.layout.clone(),
            func.call(&input.layout, &input.storage),
        )))
    }

    fn execute_reduce<S: Storage>(
        &self,
        input_id: Id,
        dim_id: Id,
        func_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>, CpuBackendError> {
        let input = self.execute_node(&expr[input_id], expr, context)?;

        let dim = match &expr[dim_id] {
            CpuBackendLanguage::Dim(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "Dim".to_string(),
                    found: format!("{:?}", &expr[dim_id]),
                });
            }
        };

        let func_lookup = match &expr[func_id] {
            CpuBackendLanguage::ReduceFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "ReduceFunc".to_string(),
                    found: format!("{:?}", &expr[func_id]),
                });
            }
        };

        let func = context
            .reduce_funcs
            .get(func_lookup.as_str())
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(func_lookup.to_string()))?;

        Ok(Arc::new(Tensor::new(
            input.layout.clone(),
            func.call(&input.layout, &input.storage, dim),
        )))
    }

    fn execute_broadcast<S: Storage>(
        &self,
        lhs_id: Id,
        rhs_id: Id,
        corrdims_id: Id,
        func_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>, CpuBackendError> {
        let lhs_input = self.execute_node(&expr[lhs_id], expr, context)?;
        let rhs_input = self.execute_node(&expr[rhs_id], expr, context)?;

        let corrdims_lookup = match &expr[corrdims_id] {
            CpuBackendLanguage::CorrespondingDims(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "CorrespondingDims".to_string(),
                    found: format!("{:?}", &expr[corrdims_id]),
                });
            }
        };

        let corrdims = context
            .corresponding_dims
            .get(&corrdims_lookup)
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(corrdims_lookup.to_string()))?;

        let func_lookup = match &expr[func_id] {
            CpuBackendLanguage::BroadcastFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "BroadcastFunc".to_string(),
                    found: format!("{:?}", &expr[func_id]),
                });
            }
        };

        let func = context
            .broadcast_funcs
            .get(func_lookup.as_str())
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(func_lookup.to_string()))?;

        Ok(Arc::new(Tensor::new(
            lhs_input.layout.broadcast(
                &rhs_input.layout,
                &lhs_input
                    .layout
                    .signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
                        &rhs_input.layout,
                        &corrdims,
                    ),
            ),
            func.call(
                &lhs_input.layout,
                &lhs_input.storage,
                &rhs_input.layout,
                &rhs_input.storage,
                &corrdims,
            ),
        )))
    }

    fn execute_fused_map_reduce<S: Storage>(
        &self,
        input_id: Id,
        map_func_id: Id,
        dim_id: Id,
        reduce_func_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>, CpuBackendError> {
        let input = self.execute_node(&expr[input_id], expr, context)?;

        let map_func_lookup = match &expr[map_func_id] {
            CpuBackendLanguage::MapFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "MapFunc".to_string(),
                    found: format!("{:?}", &expr[map_func_id]),
                });
            }
        };

        let reduce_func_lookup = match &expr[reduce_func_id] {
            CpuBackendLanguage::ReduceFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "ReduceFunc".to_string(),
                    found: format!("{:?}", &expr[reduce_func_id]),
                });
            }
        };

        let dim = match &expr[dim_id] {
            CpuBackendLanguage::Dim(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "Dim".to_string(),
                    found: format!("{:?}", &expr[dim_id]),
                });
            }
        };

        let map_func = context
            .map_funcs
            .get(map_func_lookup.as_str())
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(map_func_lookup.to_string()))?;

        let reduce_func = context
            .reduce_funcs
            .get(reduce_func_lookup.as_str())
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(reduce_func_lookup.to_string()))?;

        let mapped = map_func.call(&input.layout, &input.storage);
        Ok(Arc::new(Tensor::new(
            input
                .layout
                .reduce(input.layout.signed_dim_to_unsigned_dim(dim)),
            reduce_func.call(&input.layout, &mapped, dim),
        )))
    }

    fn execute_fused_reduce_map<S: Storage>(
        &self,
        input_id: Id,
        dim_id: Id,
        reduce_func_id: Id,
        map_func_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>, CpuBackendError> {
        let input = self.execute_node(&expr[input_id], expr, context)?;

        let map_func_lookup = match &expr[map_func_id] {
            CpuBackendLanguage::MapFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "MapFunc".to_string(),
                    found: format!("{:?}", &expr[map_func_id]),
                });
            }
        };

        let reduce_func_lookup = match &expr[reduce_func_id] {
            CpuBackendLanguage::ReduceFunc(f) => f,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "ReduceFunc".to_string(),
                    found: format!("{:?}", &expr[reduce_func_id]),
                });
            }
        };

        let dim = match &expr[dim_id] {
            CpuBackendLanguage::Dim(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "Dim".to_string(),
                    found: format!("{:?}", &expr[dim_id]),
                });
            }
        };

        let map_func = context
            .map_funcs
            .get(map_func_lookup.as_str())
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(map_func_lookup.to_string()))?;

        let reduce_func = context
            .reduce_funcs
            .get(reduce_func_lookup.as_str())
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(reduce_func_lookup.to_string()))?;

        let reduced = reduce_func.call(&input.layout, &input.storage, dim);
        Ok(Arc::new(Tensor::new(
            input
                .layout
                .reduce(input.layout.signed_dim_to_unsigned_dim(dim)),
            map_func.call(&input.layout, &reduced),
        )))
    }

    fn execute_fused_matmul<S: Storage>(
        &self,
        lhs_id: Id,
        rhs_id: Id,
        corrdims_id: Id,
        dim_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>, CpuBackendError> {
        let lhs_input = self.execute_node(&expr[lhs_id], expr, context)?;
        let rhs_input = self.execute_node(&expr[rhs_id], expr, context)?;

        let corrdims_lookup = match &expr[corrdims_id] {
            CpuBackendLanguage::CorrespondingDims(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "CorrespondingDims".to_string(),
                    found: format!("{:?}", &expr[corrdims_id]),
                });
            }
        };

        let corrdims = context
            .corresponding_dims
            .get(&corrdims_lookup)
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(corrdims_lookup.to_string()))?;

        let dim = match &expr[dim_id] {
            CpuBackendLanguage::Dim(d) => *d,
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "Dim".to_string(),
                    found: format!("{:?}", &expr[dim_id]),
                });
            }
        };

        let multiply_func = context
            .broadcast_funcs
            .get(&format!(
                "CpuMultiply({}, {} -> {})",
                std::any::type_name::<S>(),
                std::any::type_name::<S>(),
                std::any::type_name::<S>()
            ))
            .unwrap();

        let sum_func = context
            .reduce_funcs
            .get(&format!(
                "CpuSum({} -> {})",
                std::any::type_name::<S>(),
                std::any::type_name::<S>()
            ))
            .unwrap();

        let broadcasted = multiply_func.call(
            &lhs_input.layout,
            &lhs_input.storage,
            &rhs_input.layout,
            &rhs_input.storage,
            &corrdims,
        );

        let broadcasted_layout = lhs_input.layout.broadcast(
            &rhs_input.layout,
            &lhs_input
                .layout
                .signed_corresponding_dimensions_to_unsigned_corresponding_dimensions(
                    &rhs_input.layout,
                    &corrdims,
                ),
        );

        let reduced_storage = sum_func.call(&broadcasted_layout, &broadcasted, dim);

        Ok(Arc::new(Tensor::new(
            broadcasted_layout.reduce(broadcasted_layout.signed_dim_to_unsigned_dim(dim)),
            reduced_storage,
        )))
    }

    fn execute_fused_softmax<S: Storage>(
        &self,
        input_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>, CpuBackendError> {
        let input = self.execute_node(&expr[input_id], expr, context)?;

        let exp_func = context
            .map_funcs
            .get(&format!(
                "CpuExp({} -> {})",
                std::any::type_name::<S>(),
                std::any::type_name::<S>()
            ))
            .unwrap();

        let sum_func = context
            .reduce_funcs
            .get(&format!(
                "CpuSum({} -> {})",
                std::any::type_name::<S>(),
                std::any::type_name::<S>()
            ))
            .unwrap();

        let div_func = context
            .broadcast_funcs
            .get(&format!(
                "CpuDivide({}, {} -> {})",
                std::any::type_name::<S>(),
                std::any::type_name::<S>(),
                std::any::type_name::<S>()
            ))
            .unwrap();

        let exp_tensor = Arc::new(Tensor::new(
            input.layout.clone(),
            exp_func.call(&input.layout, &input.storage),
        ));
        let sum_tensor = Arc::new(Tensor::new(
            input
                .layout
                .reduce(input.layout.signed_dim_to_unsigned_dim(-1)),
            sum_func.call(&input.layout, &exp_tensor.storage, -1),
        ));
        Ok(Arc::new(Tensor::new(
            input.layout.clone(),
            div_func.call(
                &exp_tensor.layout,
                &exp_tensor.storage,
                &sum_tensor.layout,
                &sum_tensor.storage,
                &[(-2, -1)],
            ),
        )))
    }
}
