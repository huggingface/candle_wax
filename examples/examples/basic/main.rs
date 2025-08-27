use backends::cpu::CpuBackend;
use core::{
    Layout,
    backends::{
        Backend,
        map::{Map, MapFunc},
        ops::{Relu, Sum},
        reduce::{Reduce, ReduceFunc},
    },
    storage::Storage,
    tensor::Tensor,
};
use storage::cpu::CpuStorage;

fn run<S, B>(tensor: Tensor<S>) -> Tensor<S>
where
    S: Storage,
    B: Backend
        + Relu
        + Sum
        + Map<B, S, S, S::Inner, S::Inner, <B as Relu>::Relu>
        + Reduce<B, S, S, S::Inner, S::Inner, <B as Sum>::Sum>,
    <B as Relu>::Relu: MapFunc<S, S, S::Inner, S::Inner>,
    <B as Sum>::Sum: ReduceFunc<S, S, S::Inner, S::Inner>,
{
    let tensor = tensor.reduce::<B, _, _>(2, B::Sum::default());

    tensor.map::<B, _, _>(B::Relu::default())
}

fn main() {
    let tensor = Tensor::new(
        Layout::new(vec![2, 2, 2]),
        CpuStorage {
            data: vec![1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0],
        },
    );

    let result = run::<_, CpuBackend>(tensor);

    println!("Result: {:?}", result.storage.data);
}
