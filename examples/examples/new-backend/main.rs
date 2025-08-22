use core::{
    backends::{cpu::CpuBackend, op_traits::{Map, MapFunc, Relu}, Backend}, layout::Layout, numeric::Zero, storage::{cpu::{CpuDtype, CpuStorage}, Storage}, tensor::Tensor
};

struct MyNewBackend {}

impl<B, S, T, U, V, F> Map<B, S, T, U, V, F> for MyNewBackend
where
    B: Backend,
    S: Storage<Inner = U>,
    T: Storage<Inner = V>,
    F: MapFunc<S, T, U, V>,
{

    fn map(tensor: &Tensor<S, B>, f: F) -> Tensor<T, B> {
        let layout = tensor.layout.clone();
        let storage = f.call(&tensor.layout, &tensor.storage);
        Tensor::new(layout, storage)
    }
}

pub struct MyNewBackendRelu;

impl<U: CpuDtype + Zero + std::cmp::PartialOrd> MapFunc<CpuStorage<U>, CpuStorage<U>, U, U> for MyNewBackendRelu {

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
    let tensor = tensor.map(B::RELU);
    tensor
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

// pub trait MyNewDtype: Clone {}

// impl MyNewDtype for f32 {}
// impl MyNewDtype for f64 {}
// impl MyNewDtype for i32 {}
// impl MyNewDtype for i64 {}

// pub struct MyNewStorage<T: MyNewDtype> {
//     pub data: Vec<T>,
// }

// impl<T: MyNewDtype> Storage for MyNewStorage<T> {
//     type Inner = T;
// }

// pub struct MyNewRelu;

// impl<U: MyNewDtype + Zero + std::cmp::PartialOrd> MapFunc<U, U> for MyNewRelu {
//     type InputStorage<A> = MyNewStorage<U>;
//     type OutputStorage<B> = MyNewStorage<U>;

//     fn call(&self, _layout: &Layout, storage: &MyNewStorage<U>) -> MyNewStorage<U> {
//         let transformed_data: Vec<U> = storage
//             .data
//             .iter()
//             .map(|x| if *x > U::zero() { x.clone() } else { U::zero() })
//             .collect();

//         MyNewStorage {
//             data: transformed_data,
//         }
//     }
// }

// impl<T: MyNewDtype> Relu for MyNewStorage<T> {
//     type Op = MyNewRelu;
//     const OP: Self::Op = MyNewRelu;
// }

// trait ReluOp<T, S>: MapFunc<T, T, InputStorage<T> = S, OutputStorage<T> = S> {}

// impl<T, S, Op> ReluOp<T, S> for Op where Op: MapFunc<T, T, InputStorage<T> = S, OutputStorage<T> = S>
// {}

// fn run<S, T>(tensor: Tensor<S>) -> Tensor<S>
// where
//     S: Storage<Inner = T> + Relu,
//     <S as Relu>::Op: ReluOp<T, S>,
// {
//     let tensor = tensor.map(tensor.relu());
//     tensor
// }

// fn main() {
//     let tensor = Tensor {
//         layout: Layout::new(vec![2, 2, 2]),
//         storage: MyNewStorage {
//             data: vec![1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0],
//         },
//     };

//     let result = run(tensor);

//     println!("Result: {:?}", result.storage.data);
// }
