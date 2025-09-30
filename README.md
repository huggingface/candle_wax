# Candle Wax

ML playground for experimenting and developing Candle concepts.

The wax that candles are made from.

# Design Philosophy

## The Problem

Model developers and framework maintainers have fundamentally different goals:

**Model Developers** want:
- Flexibility to experiment with new architectures
- Performance that matches existing implementations
- Freedom from worrying about low-level optimization details

**Framework Maintainers** want:
- A small set of heavily optimized kernels
- Ability to quickly develop backends for new hardware
- Maintainable, portable codebases

Currently, these goals are in tension. When a new model architecture emerges, framework maintainers must spend significant engineering hours creating bespoke implementations. You can see this pattern in projects like Transformers or Candle—each new model requires substantial custom work.

## Our Solution

### Basis Operations

All tensor operations can be decomposed into a small set of basis operations that can be composed together to achieve any desired result. We define this basis set as the minimum collection of operations where every other operation can be expressed using O(1) basis operations.

#### Examples

**Matrix Multiplication** decomposes into:
- Broadcast multiply
- Reduce sum

**Softmax** decomposes into:
- Map exponential
- Reduce sum
- Broadcast division

**Attention** decomposes into:
- 2 matrix multiplies (further decomposable)
- Broadcast division
- Softmax (further decomposable)

The basis set includes operations like:
- **Map** - Apply scalar → scalar functions (e.g., ReLU, exponential)
- **Reduce** - Aggregation with an accumulator (e.g., sum, max)
- **Broadcast** - Element-wise binary operations on scalars
- **Structural operations** - Reshape, slice, permute (require no tensor value knowledge, just memory layout)
- **Spectral operations** - Convolutions, FFT, etc.

### Modular Backend Architecture

We're building a flexible system that separates concerns and enables true extensibility:

- **External Backend Development** - Create backends in separate repositories without forking the main project.
- **Separation of Storage and Execution** - Tensors and tensor storage are decoupled from backend implementation. Operations are no longer performed directly on storage, eliminating unnecessary coupling. This means tensors can live in for example NVIDIA VRAM while running different backends. For example, one optimized for computer vision, and another for transformers, without data movement.

## Benefits

### For Model Developers
Write models using familiar high-level operations. The framework automatically transforms them into basis operations under the hood—no performance penalty, no implementation burden.

### For Framework Maintainers
- **Faster backend development** - Only implement basis operations for new hardware, Spend time optimizing critical kernels rather than implementing every conceivable operation.
- **No forking required** - Develop and maintain backends in separate repositories
- **Flexible optimization** - Start with simple implementations, then optimize critical paths using rewrite rules
- **Backend composition** - Use multiple specialized backends simultaneously without moving data

### For Everyone
This approach decouples model innovation from framework implementation speed. Model developers can experiment freely while framework maintainers maintain clean, optimized codebases, and backend developers can contribute without modifying the core framework.

# Design Philosophy Old

## The Problem

Model developers and framework maintainers have fundamentally different goals:

**Model Developers** want:
- Flexibility to experiment with new architectures
- Performance that matches existing implementations
- Freedom from worrying about low-level optimization details

**Framework Maintainers** want:
- A small set of heavily optimized kernels
- Ability to quickly develop backends for new hardware
- Maintainable, portable codebases

Currently, these goals are in tension. When a new model architecture emerges, 
framework maintainers must spend significant engineering hours creating bespoke implementations. 
You can see this pattern in projects like Transformers or Candle, 
each new model requires substantial custom work.

## Our Solution: Basis Operations

All tensor operations can be decomposed into a minimal set of 
basis operations that can be composed together to achieve any desired result. 
We define this minimal basis set as the collection of operations where every 
other operation can be expressed using O(1) basis operations.

### Examples

**Matrix Multiplication** decomposes into:
- Broadcast multiply
- Reduce sum

**Softmax** decomposes into:
- Map exponential
- Reduce sum
- Broadcast division

**Attention** decomposes into:
- 2 matrix multiplies (further decomposable)
- Broadcast division
- Softmax (further decomposable)

The basis set includes operations like:
- **Map** - Apply scalar → scalar functions (e.g., ReLU, exponential)
- **Reduce** - Aggregation with an accumulator (e.g., sum, max)
- **Broadcast** - Element-wise binary operations
- **Structural operations** - Reshape, slice, permute (require no tensor value knowledge, just memory layout)
- **Spectral operations** - Convolutions, FFT 

## Benefits

### For Model Developers
Write models using familiar high-level operations. 
The framework automatically transforms them into basis operations under the 
hood—no performance penalty, no implementation burden.

### For Framework Maintainers

- **Faster backend development** - Only implement basis operations for new hardware
- **Optimization through composition** - Define fusion rules then use e-graphs to find optimal execution plans
- **Focus on what matters** - Spend time optimizing critical kernels rather than implementing every conceivable operation

Each basis operation type has well-defined constraints:

- **Map kernels**: `scalar → scalar` transformations
- **Broadcast kernels**: `(scalar, scalar) → scalar` operations
- **Reduce kernels**: `(scalar, accumulator) → accumulator` operations
- **Structural operations**: Work purely on memory layout, independent of tensor values

These constraints make implementations straightforward while maintaining full expressiveness.

### For Everyone
This approach decouples model innovation from framework implementation speed. 
Model developers can experiment freely while framework maintainers maintain a clean, 
optimized codebase.

## Old

Model developers and Framewrok Maintainers have different goals when it comes to 
Developing Tensor frameworks.

Model developers like flexibility without having to think about how model architecture
effects performance, they want to pick a model and have it perform as well as an existing model

Framewrok maintainers want a small set of heavily optimized kernels, and want to be able 
to develop new backends quickly when new technology comes out.

Model Developers make new interesting models and then Framework developers follow
and build optimized version for that specific model.

You can see this kind of relationship with transformers or candle, where everytime a new
model is many engineer hours are spent creating a bespoke script for that particular model.

Framework maintainers need to be able to create backends from scratch and experiment
with optimizations without having to implement every operation that Model Developers need,
while maintaing portability.

And model developers shoudln't have to worry about how particular operations are implemented.

All Tensor operations can really be broken down into a few core operations, I'll call
basis operations, that can be composed together, sometimes inefficiently, to achive the desired result.

We define this basis set as the minimum set of operations such that every other operation
can be described as O(1) of these basis set operations.

The main example is matrix multiply, which can mathematically be described by a
broadcast multiply and a reduce sum.

or softmax which can be described by a map exponential, a reduce sum, and a broadcast division.

Another one is attention, which can mathematically be described by 2 matrix multiplies(which can be further broken down) a broadcast division, and a softmax(which can be further broken down).

For attention, so far we've reduced it to using only broadcasts, reduces, and maps.

Some others we need are slices, reshapes, etc... and convolutions/unfolds specro-ops etc..

So in model developer land we can still use everything we know and love, but now
it can be transformed down into our basis set.

Then from the framework maintainer, if they want to create a new backend, they
only have to implement the basis ops.

Now the framework maintainer wants to optimize by fusing operations, they can create
rules for how these composite ops are defined then an e-graph handles finding the
most optimal graph. The framework maintainer then only has to focus on creating optimized
kernels for the backend ops that really matter, for example turning the broadcast and reduce
from earlier back into a matrix multiply to be fused and run all at once.

This disconnects the desires of the model developer from the ability of the 
framework maintainer to keep up.

One of the things that is sort of hand waved away in this explination is that 
there are these basis sets of ops that handle map, reduce, etc. but no gaurenteed
implementation of relu for example. but for a given map kernel there are pretty
substantial constraints on what it can do, functionally you are just defining a
function that takes in a scalar and outputs a scalar.

Or for broadcast you are defining a function that takes in 2 scalars and outputs
a scalar,

similar for reduce, but in that case one of the scalars can theoretically be an 
accumulator.

but for other operations, like permute, or slice, you don't need to have any knowlege
of the tensor at all besides the ability to make the tensor contiguous in memory.

