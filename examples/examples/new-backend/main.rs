use macros::BackendOps;

use core::{
    Layout, Tensor,
    backends::{
        Backend,
        core_ops::Map,
        map::{MapFunc, Relu},
    },
    numeric::Zero,
    storage::{
        Storage,
        cpu::{CpuDtype, CpuStorage},
    },
};

#[derive(BackendOps)]
#[backend_ops(ops = ["Map"])]
struct MyNewBackend {}

impl Backend for MyNewBackend {}

pub struct MyNewBackendRelu;

impl<U: CpuDtype + Zero + std::cmp::PartialOrd> MapFunc<CpuStorage<U>, CpuStorage<U>, U, U>
    for MyNewBackendRelu
{
    fn call(&self, _layout: &Layout, storage: &CpuStorage<U>) -> CpuStorage<U> {
        let transformed_data: Vec<U> = storage
            .data
            .iter()
            .map(|x| if *x > U::zero() { x.clone() } else { U::zero() })
            .collect();

        CpuStorage {
            data: transformed_data,
        }
    }
}

impl Relu for MyNewBackend {
    type Relu = MyNewBackendRelu;
    const RELU: Self::Relu = MyNewBackendRelu;
}

fn run<S, B>(tensor: Tensor<S, B>) -> Tensor<S, B>
where
    S: Storage,
    B: Backend,
    B: Relu + Map<B, S, S, S::Inner, S::Inner, <B as Relu>::Relu>,
    <B as Relu>::Relu: MapFunc<S, S, S::Inner, S::Inner>,
{
    tensor.map(B::RELU)
}

fn main() {
    let tensor: Tensor<_, MyNewBackend> = Tensor::new(
        Layout::new(vec![2, 2, 2]),
        CpuStorage {
            data: vec![1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0],
        },
    );

    let result = run(tensor);

    println!("Result: {:?}", result.storage.data);
}
