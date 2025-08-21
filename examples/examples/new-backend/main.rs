use core::{
    backends::{
        CpuStorage,
        op_traits::{Map, MapFunc},
    },
    layout::Layout,
    numeric::Zero,
    op_traits::Relu,
    storage::Storage,
    tensor::{Tensor, op_traits::Map as TensorMap},
};

use macros::StorageOps;

pub trait MyNewDtype: Clone {}

impl MyNewDtype for f32 {}
impl MyNewDtype for f64 {}
impl MyNewDtype for i32 {}
impl MyNewDtype for i64 {}

#[derive(StorageOps)]
#[storage_ops(ops = ["Map"])]
pub struct MyNewStorage<T: MyNewDtype> {
    pub data: Vec<T>,
}

impl<T: MyNewDtype> Storage for MyNewStorage<T> {
    type Inner = T;
}

pub struct MyNewRelu;

impl<U: MyNewDtype + Zero + std::cmp::PartialOrd> MapFunc<U, U> for MyNewRelu {
    type InputStorage<A> = MyNewStorage<U>;
    type OutputStorage<B> = MyNewStorage<U>;

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

impl<T: MyNewDtype> Relu for MyNewStorage<T> {
    type Op = MyNewRelu;
    const OP: Self::Op = MyNewRelu;
}

trait ReluOp<T, S>: MapFunc<T, T, InputStorage<T> = S, OutputStorage<T> = S> {}

impl<T, S, Op> ReluOp<T, S> for Op where Op: MapFunc<T, T, InputStorage<T> = S, OutputStorage<T> = S>
{}

fn run<S, T>(tensor: Tensor<S>) -> Tensor<S>
where
    S: Storage<Inner = T> + Relu,
    <S as Relu>::Op: ReluOp<T, S>,
{
    let tensor = tensor.map(tensor.relu());
    tensor
}

fn main() {
    let tensor = Tensor {
        layout: Layout::new(vec![2, 2, 2]),
        storage: CpuStorage {
            data: vec![1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0],
        },
    };

    let result = run(tensor);

    println!("Result: {:?}", result.storage.data);
}
