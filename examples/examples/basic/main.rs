use core::{
    backends::{
        cpu::CpuBackend,
        op_traits::{Map, MapFunc, Reduce, ReduceFunc, Relu, Sum},
    },
    layout::Layout,
    storage::{Storage, cpu::CpuStorage},
    tensor::Tensor,
};

fn run<S, B>(tensor: Tensor<S>) -> Tensor<S>
where
    S: Storage,
    B: Relu
        + Sum
        + Map<S, S, S::Inner, S::Inner, <B as Relu>::Relu>
        + Reduce<S, S, S::Inner, S::Inner, <B as Sum>::Sum>,
    <B as Relu>::Relu: MapFunc<S, S, S::Inner, S::Inner>,
    <B as Sum>::Sum: ReduceFunc<S, S, S::Inner, S::Inner>,
{
    let tensor = B::reduce(&tensor, 2, B::SUM);
    let tensor = B::map(&tensor, B::RELU);
    tensor
}

fn main() {
    let tensor = Tensor {
        layout: Layout::new(vec![2, 2, 2]),
        storage: CpuStorage {
            data: vec![1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0],
        },
    };

    let result = run::<_, CpuBackend>(tensor);

    println!("Result: {:?}", result.storage.data);
}
