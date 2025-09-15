use core::{
    Layout,
    backends::{broadcast::BroadcastFunc, map::MapFunc, reduce::ReduceFunc},
    storage::Storage,
    tensor::Tensor,
};
use egg::Id;
use num_traits::{Zero, real::Real};
use std::{
    ops::{AddAssign, Mul},
    sync::Arc,
};
use storage::cpu::{CpuDtype, CpuStorage};

use crate::{
    cpu::ops::{
        divide::CpuDivide,
        exp::CpuExp,
        matmul::{CpuMatmul, Matmul},
        sum::CpuSum,
    },
    node_executor::EggNodeExecutor,
};

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
            CpuBackendLanguage::Tensor([tensor_id, shape_id]) => {
                self.execute_tensor(*tensor_id, *shape_id, expr, context)
            }

            CpuBackendLanguage::Map([input_id, func_id, ..]) => {
                self.execute_map(*input_id, *func_id, expr, context)
            }

            CpuBackendLanguage::Reduce([input_id, func_id, dim_id, ..]) => {
                self.execute_reduce(*input_id, *func_id, *dim_id, expr, context)
            }

            CpuBackendLanguage::Broadcast([lhs_id, rhs_id, func_id, corrdims_id, ..]) => {
                self.execute_broadcast(*lhs_id, *rhs_id, *func_id, *corrdims_id, expr, context)
            }

            CpuBackendLanguage::FusedMatmul([lhs_id, rhs_id]) => {
                println!("Executing FusedMatmul");
                let context_ptr = context as *const _ as *const u8;

                unsafe {
                    if let Ok(result) = std::panic::catch_unwind(|| {
                        let cpu_context =
                            &*(context_ptr as *const CpuBackendContext<CpuStorage<f32>>);
                        self.execute_fused_matmul(*lhs_id, *rhs_id, expr, cpu_context)
                    }) && let Ok(result) = result
                    {
                        return Ok(std::mem::transmute::<
                            std::sync::Arc<core::tensor::Tensor<storage::cpu::CpuStorage<f32>>>,
                            std::sync::Arc<core::tensor::Tensor<S>>,
                        >(result));
                    }

                    panic!("FusedMatmul failed for all CpuDtype variants");
                }
            }

            CpuBackendLanguage::FusedSoftmax([input_id]) => {
                println!("Executing FusedSoftmax");
                let context_ptr = context as *const _ as *const u8;

                unsafe {
                    if let Ok(result) = std::panic::catch_unwind(|| {
                        let cpu_context =
                            &*(context_ptr as *const CpuBackendContext<CpuStorage<f32>>);
                        self.execute_fused_softmax(*input_id, expr, cpu_context)
                    }) && let Ok(result) = result
                    {
                        return Ok(std::mem::transmute::<
                            std::sync::Arc<core::tensor::Tensor<storage::cpu::CpuStorage<f32>>>,
                            std::sync::Arc<core::tensor::Tensor<S>>,
                        >(result));
                    }

                    panic!("FusedSoftmax failed for all CpuDtype variants");
                }
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
    fn execute_tensor<S: Storage>(
        &self,
        tensor_id: Id,
        _shape_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>, CpuBackendError> {
        match &expr[tensor_id] {
            CpuBackendLanguage::TensorRef(f) => context
                .tensors
                .get(&f.id)
                .cloned()
                .ok_or(CpuBackendError::TensorNotFound(f.id)),
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "TensorRef".to_string(),
                    found: format!("{:?}", &expr[tensor_id]),
                });
            }
        }
    }

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
            .get(&func_lookup.id)
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(func_lookup.to_string()))?;

        Ok(Arc::new(Tensor::new(
            input.layout.clone(),
            func.forward(&input.layout, &input.storage),
        )))
    }

    fn execute_reduce<S: Storage>(
        &self,
        input_id: Id,
        func_id: Id,
        dim_id: Id,
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
            .get(&func_lookup.id)
            .cloned()
            .ok_or_else(|| CpuBackendError::FunctionNotFound(func_lookup.to_string()))?;

        Ok(Arc::new(Tensor::new(
            input
                .layout
                .reduce(input.layout.signed_dim_to_unsigned_dim(dim) as usize),
            func.forward(&input.layout, &input.storage, dim),
        )))
    }

    fn execute_broadcast<S: Storage>(
        &self,
        lhs_id: Id,
        rhs_id: Id,
        func_id: Id,
        corrdims_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<S>,
    ) -> Result<Arc<Tensor<S>>, CpuBackendError> {
        let lhs_input = self.execute_node(&expr[lhs_id], expr, context)?;
        let rhs_input = self.execute_node(&expr[rhs_id], expr, context)?;

        let corrdims_lookup = match &expr[corrdims_id] {
            CpuBackendLanguage::CorrespondingDims(d) => d.clone(),
            _ => {
                return Err(CpuBackendError::UnexpectedNodeType {
                    expected: "CorrespondingDims".to_string(),
                    found: format!("{:?}", &expr[corrdims_id]),
                });
            }
        };

        let corrdims = corrdims_lookup.0;

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
            .get(&func_lookup.id)
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
            func.forward(
                &lhs_input.layout,
                &lhs_input.storage,
                &rhs_input.layout,
                &rhs_input.storage,
                &corrdims,
            ),
        )))
    }

    fn execute_fused_matmul<
        U: CpuDtype
            + Zero
            + std::cmp::PartialOrd
            + Mul
            + Mul<Output = U>
            + AddAssign
            + std::fmt::Debug,
    >(
        &self,
        lhs_id: Id,
        rhs_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<CpuStorage<U>>,
    ) -> Result<Arc<Tensor<CpuStorage<U>>>, CpuBackendError> {
        let lhs_input = self.execute_node(&expr[lhs_id], expr, context)?;
        let rhs_input = self.execute_node(&expr[rhs_id], expr, context)?;

        let storage = CpuMatmul.forward(
            &lhs_input.layout,
            &lhs_input.storage,
            &rhs_input.layout,
            &rhs_input.storage,
        );

        let lhs_udim_i = lhs_input.layout.signed_dim_to_unsigned_dim(-2);
        // let lhs_udim_j = lhs_input.layout.signed_dim_to_unsigned_dim(-1);
        let rhs_udim_j = rhs_input.layout.signed_dim_to_unsigned_dim(-2);
        let rhs_udim_k = rhs_input.layout.signed_dim_to_unsigned_dim(-1);

        let mut out_shape = vec![];
        out_shape.extend_from_slice(&lhs_input.layout.shape[..lhs_udim_i]);
        out_shape.extend_from_slice(&rhs_input.layout.shape[..rhs_udim_j]);
        out_shape.push(lhs_input.layout.shape[lhs_udim_i]);
        out_shape.push(rhs_input.layout.shape[rhs_udim_k]);

        let output_layout = Layout::new(out_shape);

        Ok(Arc::new(Tensor::new(output_layout, storage)))
    }

    fn execute_fused_softmax<
        U: CpuDtype + Real + std::cmp::PartialOrd + Zero + Mul + Mul<Output = U> + AddAssign,
    >(
        &self,
        input_id: Id,
        expr: &egg::RecExpr<CpuBackendLanguage>,
        context: &CpuBackendContext<CpuStorage<U>>,
    ) -> Result<Arc<Tensor<CpuStorage<U>>>, CpuBackendError> {
        let input = self.execute_node(&expr[input_id], expr, context)?;

        let exp_tensor = Arc::new(Tensor::new(
            input.layout.clone(),
            CpuExp.forward(&input.layout, &input.storage),
        ));
        let sum_tensor = Arc::new(Tensor::new(
            input
                .layout
                .reduce(input.layout.signed_dim_to_unsigned_dim(-1)),
            CpuSum.forward(&input.layout, &exp_tensor.storage, -1),
        ));
        Ok(Arc::new(Tensor::new(
            input.layout.clone(),
            CpuDivide.forward(
                &exp_tensor.layout,
                &exp_tensor.storage,
                &sum_tensor.layout,
                &sum_tensor.storage,
                &[(-2, -1)],
            ),
        )))
    }
}
