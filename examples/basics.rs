use sage::session::context::Context;
use sage::tensor::Tensor;
use sage::var::Var;

fn main() {

    // Context specifies the processor (e.g., GPU) that executes the program.
    let mut ctx = Context::new();

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


}