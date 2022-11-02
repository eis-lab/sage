use sage::session::context::Context;
use sage::tensor::Tensor;
use sage::var::{grad, Var};

fn main() {
    let mut ctx = Context::new();

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
}