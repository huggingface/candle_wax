use core::{
    backends::{
        CpuStorage,
        op_traits::{MapFunc, ReduceFunc},
    },
    layout::Layout,
    op_traits::{Relu, Sum},
    storage::Storage,
    tensor::{
        Tensor,
        op_traits::{Map, Reduce},
    },
};

trait SumOp<T, S>: ReduceFunc<T, T, InputStorage<T> = S, OutputStorage<T> = S> {}

impl<T, S, Op> SumOp<T, S> for Op where
    Op: ReduceFunc<T, T, InputStorage<T> = S, OutputStorage<T> = S>
{
}

trait ReluOp<T, S>: MapFunc<T, T, InputStorage<T> = S, OutputStorage<T> = S> {}

impl<T, S, Op> ReluOp<T, S> for Op where Op: MapFunc<T, T, InputStorage<T> = S, OutputStorage<T> = S>
{}

fn run<S, T>(tensor: Tensor<S>) -> Tensor<S>
where
    S: Storage<Inner = T> + Sum + Relu,
    <S as Sum>::Op: SumOp<T, S>,
    <S as Relu>::Op: ReluOp<T, S>,
{
    let tensor = tensor.reduce(2, tensor.sum());
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
