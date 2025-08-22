use core::{
    Layout, Tensor,
    backends::{
        Backend, CpuBackend,
        core_ops::{Map, Reduce},
        map::{MapFunc, Relu},
        reduce::{ReduceFunc, Sum},
    },
    storage::{Storage, cpu::CpuStorage},
};

fn run<S, B>(tensor: Tensor<S, B>) -> Tensor<S, B>
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
    let tensor = tensor.reduce(2, B::SUM);
    
    tensor.map(B::RELU)
}

fn main() {
    let tensor: Tensor<_, CpuBackend> = Tensor::new(
        Layout::new(vec![2, 2, 2]),
        CpuStorage {
            data: vec![1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0],
        },
    );

    let result = run(tensor);

    println!("Result: {:?}", result.storage.data);
}
