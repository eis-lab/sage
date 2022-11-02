use sage::dataset::Dataset;
use sage::dataset::mnist::Mnist;
use sage::layers::{Layer, Parameter};
use sage::models::resnet::{ResNet, ResNetConfig};
use sage::ops::metric::accuracy;
use sage::ops::nn::softmax_cross_entropy;
use sage::optim::{Adam, Optimizer};
use sage::session::context::Context;
use sage::session::device::Device;
use sage::session::Program;
use sage::tensor::data::DataType;
use sage::tensor::Tensor;
use sage::var::{grad_param, Var};

fn main() {
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
}