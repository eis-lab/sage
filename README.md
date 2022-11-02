# Sage

Sage is an experimental deep learning framework written in Rust. Sage is designed for building high-performance
differentiable programs with complex runtime logic.
Ideally, it aims to bring [PyTorch](https://pytorch.org/)-level flexibility and [TVM](https://tvm.apache.org/)-level
performance together by leveraging lazy evaluation and JIT compilation.

Core features:

- Lazy and incremental tensor evaluation
- Optimized JIT compilation (OpenCL)
- Efficient runtime memory management

**Disclaimer**: Sage is still in a very early stage of development. Numerical correctness of operation is not
guaranteed. There will be breaking API changes without prior notice.

## Installation

The core framework of Sage is written in pure Rust, but it depends on [OpenCL](https://www.khronos.org/opencl/) for GPU
support. Please check whether the system has an OpenCL driver installed.
For Android builds, it is necessary to link the OpenCL library (i.e., `libOpenCL.so`) extracted from the target platform.

## Documentation

Visit [sage.rs](https://sage.rs/) for examples and documentation (work in progress)

## Example

### Basic usage

#### Tensors and Variables

```rust
// Context specifies the processor (e.g., GPU) that executes the program.
let mut ctx = Context::with_device(2);

// Tensors are n-dimension array
let x_data = Tensor::new([
    [0.5173, -0.9896, -0.7773],
    [0.1546, -0.7499, 0.2420],
    [-1.6632, 1.0712, -0.2654],
]).to_device(&mut ctx);

// Variables hold (un)evaluated tensors.
let x = Var::new(x_data);

let y = Var::new(Tensor::new([
    [0.5173, -0.9896, -0.7773],
    [0.1546, -0.7499, 0.2420],
    [-1.6632, 1.0712, -0.2654],
]).to_device(&mut ctx));
```

#### Lazy evaluation

```rust
// New variable is created as a result of operation
// There are no actual computations at this moment
let z = &x * &y + (&x * 3.14);

// Tensor is evaluated when eval() is called
let z_data = z.eval(&mut ctx);
println!("{:?}", z_data);

// Because c already contains evaluated tensor,
// this only computes addition of the two tensors
let u_data = (&z + &x).eval(&mut ctx);
println!("{:?}", u_data);
```

#### Basic operators

```rust
// Arithmetic operators
let y = (&x * &x - &x) / &x;

// Math functions
x.abs(); x.log(); x.exp(); x.sqrt(); x.erf(); ...

// Trigonometric functions
x.sin(); x.sinh(); x.asin(); x.asinh(); ...

// Rounding functions
x.round(); x.ceil(); x.floor(); ...

// Logical operators
and(&x, &y); or(&x, &y); gt(&x, &y); le(&x, &y); ...

// Conditional operator (ternary operator)
cond(gt(&x, 0.0), &x, &y);

// Datatype casting
x.int(); x.float(); ...

```

#### Tensor shaping

```rust
// Tensor extent (i.e., shape() in NumPy)
assert_eq!(x.extents(), &[3, 3]);

// Tensor rank (i.e., ndim() in NumPy)
assert_eq!(x.rank(), 2);

// For binary operations, tensor shapes are broadcasted
// (c.f., https://numpy.org/doc/stable/user/basics.broadcasting.html)
let y = &x + Tensor::new([[1.0], [2.0], [3.0]]);

// Shape manipulations
x.transpose(0, 1);
x.permute([1, 0]);
x.unsqueeze(0).squeeze(0);
x.expand([1, 3, 3]);
x.reshape([1, 9]);
```

#### Indexing operators

```rust
// Slicing
x.slice(0, 0, 2);

// Concatenation
concat([&x, &y, &z]);

// Gather and scatter
let t = Tensor::new([
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
]);
let y = x.gather(t, 0);
let x = y.scatter(t, [3, 3]);
```

#### Reduction operators

```rust
// Summation
x.sum([0, 1], true);

// Product
x.prod(0, true);

// Minimum and maximum
x.min(0, true);
x.max(0, true);

// Example: softmax cross entropy
fn log_sum_exp(x: Var, axes: Vec<usize>) -> Var
{
    let c = x.max(&axes, true);
    (x - &c).exp().sum(&axes, true).log() + c
}

fn softmax_cross_entropy(x1: Var, x2: Var) -> Var
{
    let log_z = &x1 - log_sum_exp(&x1, 1);
    let log_p = log_z.gather(x2, 1); //log_z * x2;

    -log_p.sum(1, false)
}
```

#### Contraction operators

```rust
// Matrix multiplication
x.matmul(&y);

// Batched matrix multiplication
x.batch_matmul(&y);
```

### Automatic differentiation

All operations defined for `Variable` is differentiable. The gradient of a variable can be obtained by `grad()`
function.

```rust
let x_data = Tensor::new([
    [0.5173, -0.9896, -0.7773],
    [0.1546, -0.7499, 0.2420],
    [-1.6632, 1.0712, -0.2654],
]).to_device(&mut ctx);

// Variables hold (un)evaluated tensors.
let x = Var::new(x_data);
let y = (&x + 3.0) * (&x + 5.5);

let gy = grad(&y, [&x]);

// Get gradient of x
let gygx = gy.get(&x).unwrap();

// Higher-order differentiation is also possible
let ggygx = grad(gygx, [&x]);
let ggyggx = ggygx.get(&x).unwrap();

println!("{:?}", ggyggx.eval(&mut ctx));
```

### Neural Networks

Sage provide basic set of neural network operators required to implement basic DNN models.

#### Defining a model

Visit `src/model` for more advanced examples, such as [ResNet](https://arxiv.org/abs/1512.03385)
, [DenseNet](https://arxiv.org/abs/1608.06993), [MobileNet v2](https://arxiv.org/abs/1801.04381),
and [BERT](https://arxiv.org/abs/1810.04805).

```rust
let mut model = layers::Sequential::new();

model
.add(layers::Conv2d::new(1, 64, [3, 3]))
.add(layers::Relu)
.add(layers::MaxPool2d::new([2, 2]))
.add(layers::Conv2d::new(64, 128, [3, 3]))
.add(layers::Relu)
.add(layers::MaxPool2d::new([2, 2]))
.add(layers::Conv2d::new(128, 128, [3, 3]))
.add(layers::Relu)
.add(layers::Flatten)
.add(layers::Dense::new(3 * 3 * 128, 64))
.add(layers::Relu)
.add(layers::Dense::new(64, 10));

let logits = model.pass(&x);
```

#### Training a model

Several momentum-based optimizers (e.g., [Adam](https://arxiv.org/abs/1412.6980)) are available.

```rust
println!("{:?}", Device::get_list());

let mut ctx = Context::new();

let mut model = ResNet::new(ResNetConfig::d18(1, 10));

let batch_size = 128;
let num_epoch = 30;
let learning_rate = 1e-4;

let dataset = Mnist::from_source(
    "./dataset/mnist/train-images.idx3-ubyte",
    "./dataset/mnist/train-labels.idx1-ubyte",
).unwrap();

let mut optimizer = Adam::new(learning_rate);

model.init(&mut ctx, 0);
optimizer.init(&mut ctx);

let input = Var::empty([batch_size, 28, 28, 1], DataType::Float);
let label = Var::empty([batch_size, 1], DataType::Uint);

let logits = model.pass(&input);

let loss = softmax_cross_entropy(&logits, &label).mean(0, false);
let grads = grad_param(&loss, &model);
let acc = accuracy(&logits, &label);

let p = Program::compile(&[], grads.values().chain([&loss, &acc]));

for i in 0..num_epoch {
    for (j, (images, labels)) in dataset.iter().batch(batch_size, Mnist::collate).enumerate() {
        let (images, labels) = (images.to_device(&mut ctx), labels.to_device(&mut ctx));
        
        input.set(images);
        label.set(labels);
        
        p.exec(&mut ctx);
        
        optimizer.update(&grads, &mut ctx);
        
        println!(
            "epoch {:?} / batch {:?} / acc: {:?} / loss: {:?}",
            i,
            j,
            acc.eval(&mut ctx).to_host().scalar::<f32>(),
            loss.eval(&mut ctx).to_host().scalar::<f32>(),
        );
        
        ctx.data.clear();
    }
}
```

### Runtime memory management

Sage has several built-in tensor memory management strategies to support large-scale model training and
memory-constrained computing environments.
Please read [our paper](https://dl.acm.org/doi/abs/10.1145/3498361.3539765) on memory-efficient on-device training for
more details.

## License

Sage is licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.