use macros::BackendOps;

use core::{
    Layout,
    backends::{
        Backend, LazyBackend,
        map::{Map, MapFunc},
        ops::Relu,
    },
    storage::Storage,
    tensor::{LazyTensor, Tensor},
};
use num_traits::Zero;
use storage::cpu::{CpuDtype, CpuStorage};

#[derive(BackendOps)]
#[backend_ops(ops = ["Map"])]
struct MyNewBackend {}

impl Backend for MyNewBackend {}

impl LazyBackend for MyNewBackend {
    type LazyBackendError = ();
    fn eval<S: Storage>(tensor: LazyTensor<S>) -> Result<Tensor<S>, Self::LazyBackendError> {
        match tensor {
            LazyTensor::Tensor(t) => Ok(t.as_ref().clone()),
            LazyTensor::Map { input, func, .. } => {
                let new_tensor = Self::eval(*input)?;
                Ok(Tensor::new(
                    new_tensor.layout.clone(),
                    func.forward(&new_tensor.layout, &new_tensor.storage),
                ))
            }
            LazyTensor::Reduce {
                input, dim, func, ..
            } => {
                let new_tensor = Self::eval(*input)?;
                Ok(Tensor::new(
                    new_tensor.layout.clone(),
                    func.forward(&new_tensor.layout, &new_tensor.storage, dim),
                ))
            }
            _ => panic!("Unsupported operation in MyNewBackend"),
        }
    }
}

#[derive(Debug)]
pub struct MyNewBackendRelu;

impl<U: CpuDtype + Zero + std::cmp::PartialOrd> MapFunc<CpuStorage<U>, CpuStorage<U>, U, U>
    for MyNewBackendRelu
{
    fn forward(&self, _layout: &Layout, storage: &CpuStorage<U>) -> CpuStorage<U> {
        let transformed_data: Vec<U> = storage
            .data
            .iter()
            .map(|x| if *x > U::zero() { x.clone() } else { U::zero() })
            .collect();

        CpuStorage {
            data: transformed_data,
        }
    }

    fn hint_string(&self) -> String {
        format!(
            "MyNewBackendRelu(CpuStorage<{}> -> CpuStorage<{}>)",
            std::any::type_name::<U>(),
            std::any::type_name::<U>()
        )
    }
}

impl Default for MyNewBackendRelu {
    fn default() -> Self {
        MyNewBackendRelu
    }
}

impl Relu for MyNewBackend {
    type Relu = MyNewBackendRelu;
}

fn run<S, B>(tensor: Tensor<S>) -> Tensor<S>
where
    S: Storage,
    B: Backend,
    B: Relu + Map<B, S, S, S::Inner, S::Inner, <B as Relu>::Relu>,
    <B as Relu>::Relu: MapFunc<S, S, S::Inner, S::Inner>,
{
    tensor.map::<B, _, _>(B::Relu::default())
}

fn main() {
    let tensor = Tensor::new(
        Layout::new(vec![2, 2, 2]),
        CpuStorage {
            data: vec![1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0],
        },
    );

    let result = run::<_, MyNewBackend>(tensor);

    println!("Result: {:?}", result.storage.data);
}
