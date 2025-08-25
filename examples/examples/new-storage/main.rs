use macros::BackendOps;

use core::{
    Layout, Tensor,
    backends::{
        Backend,
        core_ops::Map,
        map::{MapFunc, Relu},
    },
    numeric::Zero,
    storage::Storage,
};

pub trait MyNewDtype: Clone {}

impl MyNewDtype for f32 {}

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
}

impl Default for MyNewBackendRelu {
    fn default() -> Self {
        MyNewBackendRelu
    }
}

impl Relu for MyNewBackend {
    type Relu = MyNewBackendRelu;
}

fn run<S, B>(tensor: Tensor<S, B>) -> Tensor<S, B>
where
    S: Storage,
    B: Backend,
    B: Relu + Map<B, S, S, S::Inner, S::Inner, <B as Relu>::Relu>,
    <B as Relu>::Relu: MapFunc<S, S, S::Inner, S::Inner>,
{
    tensor.map(B::Relu::default())
}

fn main() {
    let tensor: Tensor<_, MyNewBackend> = Tensor::new(
        Layout::new(vec![2, 2, 2]),
        MyNewStorage {
            data: vec![1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0],
        },
    );

    let result = run(tensor);

    println!("Result: {:?}", result.storage.data);
}
