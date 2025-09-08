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
use macros::BackendOps;
use num_traits::Zero;

pub trait MyNewDtype: Clone {}

impl MyNewDtype for f32 {}

#[derive(Clone)]
pub struct MyNewStorage<T: MyNewDtype> {
    pub data: Vec<T>,
}

impl<T: MyNewDtype> Storage for MyNewStorage<T> {
    type Inner = T;
}

#[derive(BackendOps)]
#[backend_ops(ops = ["Map"])]
struct MyNewBackend {}

impl Backend for MyNewBackend {}

impl LazyBackend for MyNewBackend {
    fn eval<S: Storage>(tensor: LazyTensor<S>) -> Tensor<S> {
        match tensor {
            LazyTensor::Tensor(t) => t.as_ref().clone(),
            LazyTensor::Map { input, func } => {
                let new_tensor = Self::eval(*input);
                Tensor::new(
                    new_tensor.layout.clone(),
                    func.call(&new_tensor.layout, &new_tensor.storage),
                )
            }
            LazyTensor::Reduce { input, dim, func } => {
                let new_tensor = Self::eval(*input);
                Tensor::new(
                    new_tensor.layout.clone(),
                    func.call(&new_tensor.layout, &new_tensor.storage, dim),
                )
            }
            _ => panic!("Unsupported operation in MyNewBackend"),
        }
    }
}

#[derive(Debug)]
pub struct MyNewBackendRelu;

impl<U: MyNewDtype + Zero + std::cmp::PartialOrd> MapFunc<MyNewStorage<U>, MyNewStorage<U>, U, U>
    for MyNewBackendRelu
{
    fn call(&self, _layout: &Layout, storage: &MyNewStorage<U>) -> MyNewStorage<U> {
        let transformed_data: Vec<U> = storage
            .data
            .iter()
            .map(|x| if *x > U::zero() { x.clone() } else { U::zero() })
            .collect();

        MyNewStorage {
            data: transformed_data,
        }
    }

    fn as_str(&self) -> String {
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
        MyNewStorage {
            data: vec![1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0],
        },
    );

    let result = run::<_, MyNewBackend>(tensor);

    println!("Result: {:?}", result.storage.data);
}
