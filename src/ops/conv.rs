use crate::error::Error;
use crate::ops::{NaryOperator, Operator};
use crate::session::context::{CachedAccess, Context};
use crate::shape::SizedExtent;
use crate::tensor::data::DataType;
use crate::tensor::{Tensor, TensorDesc};
use crate::var::{Var, Variable};

#[derive(Clone, Debug)]
struct Im2col<const N: usize> {
    input: [TensorDesc; 1],
    output: TensorDesc,
    col_size: [usize; N],
    ker_size: [usize; N],
    dilation: [usize; N],
    stride: [usize; N],
    padding: [usize; N],
    cached: CachedAccess,
}

#[derive(Clone, Debug)]
struct Col2im<const N: usize> {
    input: [TensorDesc; 1],
    output: TensorDesc,
    img_size: [usize; N],
    ker_size: [usize; N],
    dilation: [usize; N],
    stride: [usize; N],
    padding: [usize; N],
    cached: CachedAccess,
}

fn conv_transpose_size<const N: usize>(
    size: [usize; N],
    ker: [usize; N],
    stride: [usize; N],
    pad: [usize; N],
    dil: [usize; N],
) -> [usize; N] {
    let mut out = [0; N];
    for i in 0..N {
        out[i] = (size[i] - 1) * stride[i] - 2 * pad[i] + dil[i] * (ker[i] - 1)
    }
    out
}

fn conv_size<const N: usize>(
    size: [usize; N],
    ker: [usize; N],
    stride: [usize; N],
    pad: [usize; N],
    dil: [usize; N],
) -> [usize; N] {
    let mut out = [0; N];
    for i in 0..N {
        out[i] = (size[i] + 2 * pad[i] - dil[i] * (ker[i] - 1) - 1) / stride[i] + 1
    }
    out
}

pub fn conv_2d<V1, V2, E>(x: V1, filter: V2, stride: E, padding: E, dilation: E) -> Var
where
    V1: Variable,
    V2: Variable,
    E: SizedExtent<2>,
{
    // filter: [KH, KW, C, OC] -> [KH*KW*C, OC]
    let filter = filter.into_var();

    assert_eq!(filter.rank(), 4);

    let ker_h = filter.extent(0);
    let ker_w = filter.extent(1);
    let c_in = filter.extent(2);
    let c_out = filter.extent(3);

    let stride = stride.to_arr();
    let padding = padding.to_arr();
    let dilation = dilation.to_arr();

    // image: [N, OH, OW, KH, KW, C] -> [N*OH*OW, KH*KW*C]

    let col = im2col(x, [ker_w, ker_h], stride, padding, dilation);

    assert_eq!(col.extent(5), c_in);

    let n = col.extents()[0];
    let col_h = col.extents()[1];
    let col_w = col.extents()[2];

    // image * filter =  [N*OH*OW, KH*KW*C] * [KH*KW*C, OC] -> [N*OH*OW, OC] -> [N, OH, OW, OC]
    let col = col.view([n * col_h * col_w, ker_h * ker_w * c_in]);
    let filter = filter.view([ker_h * ker_w * c_in, c_out]);

    col.matmul(filter).view([n, col_h, col_w, c_out])
}

pub fn conv_transpose_2d<V1, V2, E>(x: V1, filter: V2, stride: E, padding: E, dilation: E) -> Var
where
    V1: Variable,
    V2: Variable,
    E: SizedExtent<2>,
{
    //(KH, KW, C, OC)
    let filter = filter.into_var();

    // (N, H, W, C)
    let x = x.into_var();

    let n = x.extents()[0];
    let img_h = x.extents()[1];
    let img_w = x.extents()[2];
    let c_in = x.extents()[3];
    let c_out = filter.extents()[2];
    let ker_h = filter.extents()[0];
    let ker_w = filter.extents()[1];

    // (N, H, W, C) -> (N*H*W, C)
    let x = x.view([n * img_h * img_w, c_in]);

    // (KH, KW, C, OC) -> (C, KH, KW, OC) -> (C, KH*KW*OC)
    let filter = filter
        .permute([2, 0, 1, 3])
        .organize()
        .view([c_in, ker_h * ker_w * c_out]);

    // (N*H*W, C) * (C, KH*KW*OC) -> (N*H*W, KH*KW*OC) -> (N, H, W, KH, KW, OC)
    let col = x
        .matmul(filter)
        .view([n, img_h, img_w, ker_h, ker_w, c_out]);

    let ker_size = [ker_w, ker_h];
    let stride = stride.to_arr();
    let padding = padding.to_arr();
    let dilation = dilation.to_arr();

    let out = conv_transpose_size([img_w, img_h], ker_size, stride, padding, dilation);

    // (N, H, W, KH, KW, OC) -> (N, OH, OW, OC)
    col2im(col, out, ker_size, stride, padding, dilation)
}

pub fn avg_pool_2d<V, E>(x: V, ker_size: E, stride: E, padding: E, dilation: E) -> Var
where
    V: Variable,
    E: SizedExtent<2>,
{
    // img (N, H, W, C) - > col (N, OH, OW, KH, KW, C)
    let col = im2col(x, ker_size, stride, padding, dilation);

    // col (N, OH, OW, KH, KW, C) -> out (N, OH, OW, C)
    col.mean([3, 4], false)
}

pub fn max_pool_2d<V, E>(x: V, ker_size: E, stride: E, padding: E, dilation: E) -> Var
where
    V: Variable,
    E: SizedExtent<2>,
{
    // img (N, H, W, C) - > col (N, OH, OW, KH, KW, C)
    let col = im2col(x, ker_size, stride, padding, dilation);

    // col (N, OH, OW, KH, KW, C) -> out (N, OH, OW, C)
    col.max([3, 4], false)
}

pub fn batch_norm_2d<V1, V2, V3>(x: V1, gamma: V2, beta: V2, mean: V3, var: V3, eps: f32) -> Var
where
    V1: Variable,
    V2: Variable,
    V3: Variable,
{
    let x = x.into_var();
    let gamma = gamma.into_var();
    let beta = beta.into_var();
    let mean = mean.into_var();
    let var = var.into_var();

    //let mean = x.mean([0, 1, 2], true);
    //let var = x.var([0, 1, 2], true);

    let xc = (x - mean) / (var + eps).sqrt();

    gamma * xc + beta
}

// [N, H, W, C] -> [N, OH, OW, KH, KW, C]
pub fn im2col<V, E>(x: V, ker_size: E, stride: E, padding: E, dilation: E) -> Var
where
    V: Variable,
    E: SizedExtent<2>,
{
    let x = x.into_var().organize();
    let ker_size = ker_size.to_arr();
    let stride = stride.to_arr();
    let padding = padding.to_arr();
    let dilation = dilation.to_arr();

    assert_eq!(x.rank(), 4);
    assert_eq!(x.shape().has_default_strides(), true);

    let n = x.extent(0);
    let chan = x.extent(3);

    let img_size = [x.extent(2), x.extent(1)];
    let col_size = conv_size(img_size, ker_size, stride, padding, dilation);

    let output = TensorDesc::new(
        [n, col_size[1], col_size[0], ker_size[1], ker_size[0], chan],
        x.data_type(),
    );

    Var::from_unary_op(
        Im2col {
            input: [x.desc().clone()],
            output,
            col_size,
            ker_size,
            dilation,
            stride,
            padding,
            cached: Default::default(),
        },
        x,
    )
}

// [N, OH, OW, KH, KW, C] -> [N, H, W, C]
pub fn col2im<V, E>(
    x: V,
    img_size: [usize; 2],
    ker_size: E,
    stride: E,
    padding: E,
    dilation: E,
) -> Var
where
    V: Variable,
    E: SizedExtent<2>,
{
    let x = x.into_var().organize();

    assert_eq!(x.rank(), 6);
    assert_eq!(x.shape().has_default_strides(), true);

    let n = x.extent(0);
    let chan = x.extent(5);

    let output = TensorDesc::new([n, img_size[1], img_size[0], chan], x.data_type());

    Var::from_unary_op(
        Col2im {
            input: [x.desc().clone()],
            output,
            img_size,
            ker_size: ker_size.to_arr(),
            dilation: dilation.to_arr(),
            stride: stride.to_arr(),
            padding: padding.to_arr(),
            cached: Default::default(),
        },
        x,
    )
}

impl NaryOperator<1> for Im2col<2> {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Var; 1], _: &Var, gy: &Var) -> [Option<Var>; 1] {
        let img_h = x[0].extent(1);
        let img_w = x[0].extent(2);

        [Some(col2im(
            gy,
            [img_w, img_h],
            self.ker_size,
            self.stride,
            self.padding,
            self.dilation,
        ))]
    }

    fn compute(&self, x: [&Tensor; 1], ctx: &mut Context) -> Result<Tensor, Error> {
        let x = x[0];

        let n = x.extent(0);
        let img_h = x.extent(1);
        let img_w = x.extent(2);
        let chan = x.extent(3);

        // [N, H, W, C] -> [N, OH, OW, KH, KW, C]
        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        self.cached.load_program(ctx, || {
            let data_type = x.data_type().opencl();

            format!(r#"
        __kernel void im2col(__global const {data_type} *img,
            __private const int img_offset,
            __global {data_type} *col,
            __private const int col_offset,
            __private const int batch,
            __private const int chan,
            __private const int img_w, __private const int img_h,
            __private const int col_w, __private const int col_h,
            __private const int ker_w, __private const int ker_h,
            __private const int stride_w, __private const int stride_h,
            __private const int padding_w, __private const int padding_h,
            __private const int dilation_w, __private const int dilation_h
        ) {{

            const int gid = get_global_id(0);

            for (int n = 0; n < batch; ++n) {{

                const int offset_img_batch = n * img_h * img_w * chan;
                const int offset_col_batch = n * col_h * col_w * ker_h * ker_w * chan;

                const int c = gid % chan;
                const int col_x = (gid / chan) % col_w;
                const int col_y = gid / (chan * col_w);

                const int img_x_base = col_x * stride_w - padding_w;
                const int img_y_base = col_y * stride_h - padding_h;

                for (int ker_y = 0; ker_y < ker_h; ++ker_y) {{
                    for (int ker_x = 0; ker_x < ker_w; ++ker_x) {{

                        int img_y = img_y_base + ker_y * dilation_h;
                        int img_x = img_x_base + ker_x * dilation_w;
                        int offset_col = offset_col_batch + (((col_y * col_w + col_x) * ker_h + ker_y) * ker_w + ker_x) * chan + col_offset;

                        float img_val = 0;

                        if (img_x >= 0 && img_x < img_w && img_y >= 0 && img_y < img_h) {{
                            int offset_img = offset_img_batch + (img_y * img_w + img_x) * chan + img_offset;
                            img_val = img[offset_img + c];
                        }}
                        col[offset_col + c] = img_val;
                    }}
                }}
            }}
        }}"#)
        }).kernel("im2col")
            .arg_tensor(x)
            .arg(x.offset() as i32)
            .arg_tensor(&y)
            .arg(y.offset() as i32)
            .arg(n as i32)
            .arg(chan as i32)
            .arg(img_w as i32)
            .arg(img_h as i32)
            .arg(self.col_size[0] as i32)
            .arg(self.col_size[1] as i32)
            .arg(self.ker_size[0] as i32)
            .arg(self.ker_size[1] as i32)
            .arg(self.stride[0] as i32)
            .arg(self.stride[1] as i32)
            .arg(self.padding[0] as i32)
            .arg(self.padding[1] as i32)
            .arg(self.dilation[0] as i32)
            .arg(self.dilation[1] as i32)
            .global_work_size(self.col_size[0] * self.col_size[1] * chan)
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }
}

impl NaryOperator<1> for Col2im<2> {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Var; 1], _: &Var, gy: &Var) -> [Option<Var>; 1] {
        // println!("x{:?}", x.extents());
        // println!("gy{:?}", gy.extents());
        // println!("gx(expect) {:?}", self.input.extents());
        // println!(
        //     "gx(actual) {:?}",
        //     im2col(gy, self.ker_size, self.stride, self.padding, self.dilation,).extents()
        // );

        [Some(im2col(
            gy,
            self.ker_size,
            self.stride,
            self.padding,
            self.dilation,
        ))]
    }

    fn compute(&self, x: [&Tensor; 1], ctx: &mut Context) -> Result<Tensor, Error> {
        let x = x[0];
        let n = x.extent(0);
        let col_h = x.extent(1);
        let col_w = x.extent(2);
        let chan = x.extent(5);

        // [N, OH, OW, KH, KW, C] -> [N, H, W, C]
        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        self.cached.load_program(ctx, || {
            let data_type = x.data_type().opencl();

            format!(r#"
        __kernel void col2im(__global const {data_type} *col,
                    __private const int col_offset,
                    __global {data_type} *img,
                    __private const int img_offset,
                    __private const int batch,
                    __private const int chan,
                    __private const int img_w, __private const int img_h,
                    __private const int col_w, __private const int col_h,
                    __private const int ker_w, __private const int ker_h,
                    __private const int stride_w, __private const int stride_h,
                    __private const int padding_w, __private const int padding_h,
                    __private const int dilation_w, __private const int dilation_h
        ) {{
            const int gid = get_global_id(0);

            for (int n = 0; n < batch; ++n) {{

                const int offset_img_batch = n * img_h * img_w * chan;
                const int offset_col_batch = n * col_h * col_w * ker_h * ker_w * chan;

                const int c = gid % chan;
                const int img_x = (gid / chan) % img_w;
                const int img_y = gid / (chan * img_w);

                const int img_x_pad = img_x + padding_w;
                const int img_y_pad = img_y + padding_h;

                const int ker_ew = (ker_w - 1) * dilation_w + 1;
                const int ker_eh = (ker_h - 1) * dilation_h + 1;

                const int col_x_start = (img_x_pad < ker_ew) ? 0 : (img_x_pad - ker_ew) / stride_w + 1;
                const int col_y_start = (img_y_pad < ker_eh) ? 0 : (img_y_pad - ker_eh) / stride_h + 1;

                const int col_x_end = min(img_x_pad / stride_w + 1, col_w);
                const int col_y_end = min((img_y + padding_h) / stride_h + 1, col_h);

                {data_type} val = 0;
                const int offset_img = (img_y * img_w + img_x) * chan + img_offset;

                for (int col_x = col_x_start; col_x < col_x_end; col_x += 1) {{
                    for (int col_y = col_y_start; col_y < col_y_end; col_y += 1) {{
                        int ker_x = img_x_pad - col_x * stride_w;
                        int ker_y = img_y_pad - col_y * stride_h;

                        if (ker_x % dilation_h == 0 && ker_y % dilation_w == 0) {{
                            ker_x /= dilation_h;
                            ker_y /= dilation_w;

                            const int offset_col = (((col_y * col_w + col_x) * ker_h + ker_y) * ker_w + ker_x) * chan + col_offset;
                            val += col[offset_col_batch + offset_col + c];
                        }}
                    }}
                }}
                img[offset_img_batch + offset_img + c] = val;
            }}
        }}"#)
        }).kernel("col2im")
            .arg_tensor(x)
            .arg(x.offset() as i32)
            .arg_tensor(&y)
            .arg(y.offset() as i32)
            .arg(n as i32)
            .arg(chan as i32)
            .arg(self.img_size[0] as i32)
            .arg(self.img_size[1] as i32)
            .arg(col_w as i32)
            .arg(col_h as i32)
            .arg(self.ker_size[0] as i32)
            .arg(self.ker_size[1] as i32)
            .arg(self.stride[0] as i32)
            .arg(self.stride[1] as i32)
            .arg(self.padding[0] as i32)
            .arg(self.padding[1] as i32)
            .arg(self.dilation[0] as i32)
            .arg(self.dilation[1] as i32)
            .global_work_size(self.img_size[0] * self.img_size[1] * chan)
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::conv::{col2im, conv_2d, im2col};
    use crate::session::context::Context;
    use crate::tensor::Tensor;
    use crate::var::{grad_check, Var};

    #[test]
    fn test_im2col() {
        let mut ctx = Context::new();

        /* *** test data generation (Pytorch) ***

            import torch
            import torch.nn as nn

            n = 1
            c = 3
            h = 5
            w = 4
            kh = 2
            kw = 3
            oc = 2

            oh = 4
            ow = 2
            image = torch.randn(n, c, h, w) # (N, C, H, W)

            print(image.permute(0, 2, 3, 1))

            # (N, C, H, W) -> (N, C*KH*KW, OH*OW)
            col = torch.nn.functional.unfold(image, (kh, kw))
            col = col.view(n, c, kh, kw, oh, ow)

            print(col.permute(0, 4, 5, 2, 3, 1)) # (N, H, W, K, K, C)
            # (N, C, K, K, H, W) -> (N, H, W, K, K, C)
            col = col.view(n, c * kh * kw, oh * ow)

            output = torch.nn.functional.fold(col, output_size=(h, w), kernel_size=(kh, kw))

            # (N, C, H, W) -> (N, H, W, C)
            print(output.permute(0, 2, 3, 1))
        */

        let x = Tensor::new([[
            [
                [-0.3912, 0.2274, 0.0326],
                [0.8053, 0.7159, -0.1222],
                [0.7102, 0.5819, 0.3437],
                [-0.2192, 0.4577, -0.2126],
            ],
            [
                [-0.1271, -0.9265, 0.0657],
                [-0.6882, -1.8182, -0.4486],
                [0.4815, 1.0864, 1.9958],
                [-0.0351, 0.0710, -0.4102],
            ],
            [
                [-2.5302, 1.0427, 1.8720],
                [-0.1881, -3.0820, -1.0174],
                [-0.6383, -1.3758, -1.2978],
                [0.8032, -0.3860, 1.6866],
            ],
            [
                [0.2761, -1.0947, 0.8361],
                [0.2724, 0.1792, 0.5878],
                [-0.1513, 0.0232, -0.5367],
                [2.2823, 0.5321, -1.8185],
            ],
            [
                [-0.2796, 1.0467, 1.2866],
                [-0.0163, -1.1627, 1.1465],
                [0.1353, -0.8894, 0.9887],
                [1.4485, 0.5912, -0.9649],
            ],
        ]])
        .to_device(&mut ctx);

        //[N, OH, OW, KH, KW, C]
        let y_gt = Tensor::new([[
            [
                [
                    [
                        [-0.3912, 0.2274, 0.0326],
                        [0.8053, 0.7159, -0.1222],
                        [0.7102, 0.5819, 0.3437],
                    ],
                    [
                        [-0.1271, -0.9265, 0.0657],
                        [-0.6882, -1.8182, -0.4486],
                        [0.4815, 1.0864, 1.9958],
                    ],
                ],
                [
                    [
                        [0.8053, 0.7159, -0.1222],
                        [0.7102, 0.5819, 0.3437],
                        [-0.2192, 0.4577, -0.2126],
                    ],
                    [
                        [-0.6882, -1.8182, -0.4486],
                        [0.4815, 1.0864, 1.9958],
                        [-0.0351, 0.0710, -0.4102],
                    ],
                ],
            ],
            [
                [
                    [
                        [-0.1271, -0.9265, 0.0657],
                        [-0.6882, -1.8182, -0.4486],
                        [0.4815, 1.0864, 1.9958],
                    ],
                    [
                        [-2.5302, 1.0427, 1.8720],
                        [-0.1881, -3.0820, -1.0174],
                        [-0.6383, -1.3758, -1.2978],
                    ],
                ],
                [
                    [
                        [-0.6882, -1.8182, -0.4486],
                        [0.4815, 1.0864, 1.9958],
                        [-0.0351, 0.0710, -0.4102],
                    ],
                    [
                        [-0.1881, -3.0820, -1.0174],
                        [-0.6383, -1.3758, -1.2978],
                        [0.8032, -0.3860, 1.6866],
                    ],
                ],
            ],
            [
                [
                    [
                        [-2.5302, 1.0427, 1.8720],
                        [-0.1881, -3.0820, -1.0174],
                        [-0.6383, -1.3758, -1.2978],
                    ],
                    [
                        [0.2761, -1.0947, 0.8361],
                        [0.2724, 0.1792, 0.5878],
                        [-0.1513, 0.0232, -0.5367],
                    ],
                ],
                [
                    [
                        [-0.1881, -3.0820, -1.0174],
                        [-0.6383, -1.3758, -1.2978],
                        [0.8032, -0.3860, 1.6866],
                    ],
                    [
                        [0.2724, 0.1792, 0.5878],
                        [-0.1513, 0.0232, -0.5367],
                        [2.2823, 0.5321, -1.8185],
                    ],
                ],
            ],
            [
                [
                    [
                        [0.2761, -1.0947, 0.8361],
                        [0.2724, 0.1792, 0.5878],
                        [-0.1513, 0.0232, -0.5367],
                    ],
                    [
                        [-0.2796, 1.0467, 1.2866],
                        [-0.0163, -1.1627, 1.1465],
                        [0.1353, -0.8894, 0.9887],
                    ],
                ],
                [
                    [
                        [0.2724, 0.1792, 0.5878],
                        [-0.1513, 0.0232, -0.5367],
                        [2.2823, 0.5321, -1.8185],
                    ],
                    [
                        [-0.0163, -1.1627, 1.1465],
                        [0.1353, -0.8894, 0.9887],
                        [1.4485, 0.5912, -0.9649],
                    ],
                ],
            ],
        ]])
        .to_device(&mut ctx);

        let x = Var::new(x);
        let y = im2col(&x, [3, 2], [1, 1], [0, 0], [1, 1]);

        // let c = gemm(&a, &b, true, true).eval(&mut ctx);
        assert!(Tensor::all_close(&y.eval(&mut ctx), &y_gt, 0.001));

        // conv grad
        assert!(grad_check(&y, &x, 0.01, &mut ctx));
    }

    #[test]
    fn test_col2im() {
        let mut ctx = Context::new();

        // [N, OH, OW, KH, KW, C] -> [N, H, W, C]
        let x = Tensor::new([
            [
                [
                    [
                        [
                            [-0.7271, -1.0278, 0.5188],
                            [1.3826, 0.1245, 1.9218],
                            [0.4845, -0.2310, -0.4660],
                        ],
                        [
                            [1.4023, 0.0921, -0.3564],
                            [-0.7691, 0.4323, 1.2341],
                            [0.6568, -0.6111, 0.1896],
                        ],
                    ],
                    [
                        [
                            [1.3826, 0.1245, 1.9218],
                            [0.4845, -0.2310, -0.4660],
                            [-0.1042, 1.1151, 0.0839],
                        ],
                        [
                            [-0.7691, 0.4323, 1.2341],
                            [0.6568, -0.6111, 0.1896],
                            [1.1807, 1.4796, 0.2908],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [1.4023, 0.0921, -0.3564],
                            [-0.7691, 0.4323, 1.2341],
                            [0.6568, -0.6111, 0.1896],
                        ],
                        [
                            [0.2706, -0.6863, -0.0958],
                            [0.1823, -0.1452, 0.6225],
                            [-1.3960, -2.0233, -0.4229],
                        ],
                    ],
                    [
                        [
                            [-0.7691, 0.4323, 1.2341],
                            [0.6568, -0.6111, 0.1896],
                            [1.1807, 1.4796, 0.2908],
                        ],
                        [
                            [0.1823, -0.1452, 0.6225],
                            [-1.3960, -2.0233, -0.4229],
                            [-0.2164, -0.7825, -1.1237],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [0.2706, -0.6863, -0.0958],
                            [0.1823, -0.1452, 0.6225],
                            [-1.3960, -2.0233, -0.4229],
                        ],
                        [
                            [1.0629, -0.5537, -1.2804],
                            [2.2984, 0.0304, 0.5218],
                            [0.8047, 0.1628, 0.7657],
                        ],
                    ],
                    [
                        [
                            [0.1823, -0.1452, 0.6225],
                            [-1.3960, -2.0233, -0.4229],
                            [-0.2164, -0.7825, -1.1237],
                        ],
                        [
                            [2.2984, 0.0304, 0.5218],
                            [0.8047, 0.1628, 0.7657],
                            [0.6863, 1.9757, -1.2644],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [1.0629, -0.5537, -1.2804],
                            [2.2984, 0.0304, 0.5218],
                            [0.8047, 0.1628, 0.7657],
                        ],
                        [
                            [0.9516, 0.3332, -1.7579],
                            [0.3565, -0.1854, -1.3845],
                            [0.6383, 1.8545, 0.0617],
                        ],
                    ],
                    [
                        [
                            [2.2984, 0.0304, 0.5218],
                            [0.8047, 0.1628, 0.7657],
                            [0.6863, 1.9757, -1.2644],
                        ],
                        [
                            [0.3565, -0.1854, -1.3845],
                            [0.6383, 1.8545, 0.0617],
                            [-0.3269, 0.0394, 0.8206],
                        ],
                    ],
                ],
            ],
            [
                [
                    [
                        [
                            [-0.1320, -0.2757, -1.3143],
                            [0.4849, 1.7280, 1.3186],
                            [0.0916, 1.1564, -1.0789],
                        ],
                        [
                            [-0.0643, 1.0181, 0.6267],
                            [-0.2418, -0.3564, -1.1930],
                            [0.9575, -0.4932, -1.0130],
                        ],
                    ],
                    [
                        [
                            [0.4849, 1.7280, 1.3186],
                            [0.0916, 1.1564, -1.0789],
                            [0.0159, 0.5378, 0.4068],
                        ],
                        [
                            [-0.2418, -0.3564, -1.1930],
                            [0.9575, -0.4932, -1.0130],
                            [-0.8167, 1.8140, 0.2024],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [-0.0643, 1.0181, 0.6267],
                            [-0.2418, -0.3564, -1.1930],
                            [0.9575, -0.4932, -1.0130],
                        ],
                        [
                            [-0.5481, 1.5046, 1.0490],
                            [1.5011, -0.6098, 0.7700],
                            [0.8107, 1.0728, 1.9587],
                        ],
                    ],
                    [
                        [
                            [-0.2418, -0.3564, -1.1930],
                            [0.9575, -0.4932, -1.0130],
                            [-0.8167, 1.8140, 0.2024],
                        ],
                        [
                            [1.5011, -0.6098, 0.7700],
                            [0.8107, 1.0728, 1.9587],
                            [-1.1903, -0.2139, 0.6860],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [-0.5481, 1.5046, 1.0490],
                            [1.5011, -0.6098, 0.7700],
                            [0.8107, 1.0728, 1.9587],
                        ],
                        [
                            [-1.4197, 0.3093, 0.2028],
                            [-0.8521, -1.6478, 0.3162],
                            [-0.1969, 1.2813, 0.2715],
                        ],
                    ],
                    [
                        [
                            [1.5011, -0.6098, 0.7700],
                            [0.8107, 1.0728, 1.9587],
                            [-1.1903, -0.2139, 0.6860],
                        ],
                        [
                            [-0.8521, -1.6478, 0.3162],
                            [-0.1969, 1.2813, 0.2715],
                            [1.0036, -0.2979, 0.8970],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [-1.4197, 0.3093, 0.2028],
                            [-0.8521, -1.6478, 0.3162],
                            [-0.1969, 1.2813, 0.2715],
                        ],
                        [
                            [0.9969, -1.1901, -0.3415],
                            [1.0757, -0.6598, -0.2246],
                            [-1.5426, 0.4510, 1.3336],
                        ],
                    ],
                    [
                        [
                            [-0.8521, -1.6478, 0.3162],
                            [-0.1969, 1.2813, 0.2715],
                            [1.0036, -0.2979, 0.8970],
                        ],
                        [
                            [1.0757, -0.6598, -0.2246],
                            [-1.5426, 0.4510, 1.3336],
                            [0.3905, -0.7949, 1.8356],
                        ],
                    ],
                ],
            ],
        ])
        .to_device(&mut ctx);

        let y_gt = Tensor::new([
            [
                [
                    [-0.7271, -1.0278, 0.5188],
                    [2.7652, 0.2490, 3.8437],
                    [0.9690, -0.4621, -0.9321],
                    [-0.1042, 1.1151, 0.0839],
                ],
                [
                    [2.8046, 0.1842, -0.7127],
                    [-3.0762, 1.7294, 4.9365],
                    [2.6272, -2.4443, 0.7584],
                    [2.3614, 2.9593, 0.5815],
                ],
                [
                    [0.5411, -1.3725, -0.1917],
                    [0.7294, -0.5808, 2.4901],
                    [-5.5839, -8.0933, -1.6916],
                    [-0.4328, -1.5651, -2.2473],
                ],
                [
                    [2.1257, -1.1074, -2.5609],
                    [9.1935, 0.1217, 2.0873],
                    [3.2188, 0.6511, 3.0628],
                    [1.3727, 3.9514, -2.5289],
                ],
                [
                    [0.9516, 0.3332, -1.7579],
                    [0.7129, -0.3707, -2.7690],
                    [1.2766, 3.7090, 0.1234],
                    [-0.3269, 0.0394, 0.8206],
                ],
            ],
            [
                [
                    [-0.1320, -0.2757, -1.3143],
                    [0.9698, 3.4560, 2.6372],
                    [0.1832, 2.3128, -2.1578],
                    [0.0159, 0.5378, 0.4068],
                ],
                [
                    [-0.1286, 2.0362, 1.2533],
                    [-0.9673, -1.4256, -4.7719],
                    [3.8301, -1.9728, -4.0520],
                    [-1.6334, 3.6280, 0.4048],
                ],
                [
                    [-1.0963, 3.0091, 2.0980],
                    [6.0046, -2.4394, 3.0799],
                    [3.2430, 4.2912, 7.8347],
                    [-2.3806, -0.4277, 1.3720],
                ],
                [
                    [-2.8394, 0.6185, 0.4056],
                    [-3.4084, -6.5914, 1.2649],
                    [-0.7878, 5.1253, 1.0861],
                    [2.0071, -0.5958, 1.7940],
                ],
                [
                    [0.9969, -1.1901, -0.3415],
                    [2.1514, -1.3196, -0.4492],
                    [-3.0851, 0.9020, 2.6671],
                    [0.3905, -0.7949, 1.8356],
                ],
            ],
        ]);

        let x = Var::new(x);
        let y = col2im(&x, [4, 5], [3, 2], [1, 1], [0, 0], [1, 1]);

        assert!(Tensor::all_close(&y.eval(&mut ctx), &y_gt, 0.001));
        // conv grad
        assert!(grad_check(&y, &x, 0.01, &mut ctx));
    }

    #[test]
    pub fn test_conv2d() {
        // image: [N, H, W, C]
        // filter: [KH, KW, C, OC] -> [KH*KW*C, OC]

        /*
        # (OC, C, KH, KW)
        filters = torch.randn(8, 4, 3, 3)
        # (N, C, H, W)
        inputs = torch.randn(1, 4, 5, 5)
        output = F.conv2d(inputs, filters, padding=1)

        print(inputs.permute(0, 2, 3, 1))
        print(output.permute(0, 2, 3, 1))
        print(filters.permute(2, 3, 1, 0))

         */

        let mut ctx = Context::new();

        let x = Tensor::new([[
            [
                [-0.9664, 0.9046, -1.3823, 0.8345],
                [0.2371, 0.8352, -0.4215, -1.2925],
                [-0.7105, 0.1348, -1.5621, -0.0922],
                [-2.2273, -1.5452, 0.3850, 0.2842],
                [-0.2438, 0.6608, 0.0538, 1.6100],
            ],
            [
                [0.5379, -0.1161, 0.0130, 0.0747],
                [0.5191, -0.6924, -0.7166, -1.0597],
                [-0.2716, 0.7937, -1.0394, -0.3062],
                [0.9000, -1.2641, 0.6944, -0.0063],
                [0.4729, -0.1122, -0.0762, -1.7202],
            ],
            [
                [0.0525, 0.0740, -0.6805, 0.7901],
                [-1.9947, -1.0000, -0.5139, 0.4825],
                [-2.1136, 1.4135, 0.1767, 3.1634],
                [-1.2708, 0.3655, 0.4587, 0.8529],
                [-0.6193, -0.8744, -1.3453, -0.1093],
            ],
            [
                [1.1030, -0.2216, -1.1084, -1.8138],
                [-0.4808, 0.0138, 0.2072, -0.6409],
                [-0.6337, -0.0986, 0.0127, -0.7021],
                [0.4316, -0.5224, 0.5617, 0.1486],
                [-0.7335, -1.2610, -0.9921, -0.2403],
            ],
            [
                [0.3466, -0.0341, 1.5294, 1.6268],
                [1.9429, -0.2657, 1.5307, 0.1328],
                [-0.9864, -1.2792, 0.5200, -0.5477],
                [0.2173, -0.7708, -0.4076, -2.3475],
                [0.2892, -0.3749, 0.2742, -1.6684],
            ],
        ]])
        .to_device(&mut ctx);

        let filter = Tensor::new([
            [
                [
                    [
                        -2.4455e-01,
                        2.4538e-01,
                        -1.6617e-01,
                        -3.0274e-01,
                        4.2391e-01,
                        1.6966e+00,
                        -4.8545e-01,
                        1.6841e+00,
                    ],
                    [
                        -1.0770e+00,
                        2.2385e-01,
                        -1.2766e+00,
                        5.0057e-01,
                        7.0227e-01,
                        1.1719e-01,
                        -1.2494e+00,
                        -1.2618e+00,
                    ],
                    [
                        -1.1880e+00,
                        1.3946e+00,
                        -9.8764e-01,
                        7.3302e-01,
                        9.7824e-02,
                        7.2465e-02,
                        2.0041e-01,
                        2.2842e-01,
                    ],
                    [
                        1.7023e+00,
                        -1.6289e+00,
                        2.0534e+00,
                        1.1260e+00,
                        -1.2642e-01,
                        1.9643e+00,
                        -9.6251e-01,
                        4.1596e-01,
                    ],
                ],
                [
                    [
                        -5.7814e-01,
                        -1.5415e+00,
                        6.8514e-01,
                        9.3921e-01,
                        1.6818e-01,
                        -2.0013e+00,
                        3.7795e-01,
                        1.3430e+00,
                    ],
                    [
                        -1.0861e+00,
                        1.5803e-01,
                        -1.2295e+00,
                        -1.0673e+00,
                        8.4821e-01,
                        1.4156e-01,
                        1.1245e+00,
                        -4.3322e-01,
                    ],
                    [
                        7.0289e-01,
                        -3.2388e-01,
                        -8.8566e-01,
                        5.7142e-01,
                        -4.8393e-01,
                        -1.5126e+00,
                        -2.8755e-01,
                        1.1997e+00,
                    ],
                    [
                        -1.3055e+00,
                        2.6570e-01,
                        -4.7839e-01,
                        -6.1484e-01,
                        9.2818e-02,
                        -1.2081e+00,
                        6.0014e-01,
                        5.4400e-01,
                    ],
                ],
                [
                    [
                        -1.7832e-01,
                        1.5284e-01,
                        -3.0294e-01,
                        -1.6614e+00,
                        4.4800e-02,
                        -8.6474e-01,
                        6.0318e-01,
                        3.6825e-01,
                    ],
                    [
                        -1.6306e+00,
                        -7.2887e-02,
                        -8.5033e-01,
                        -2.8331e-01,
                        -1.0776e+00,
                        2.6355e+00,
                        -1.3058e+00,
                        -4.9751e-01,
                    ],
                    [
                        2.8795e+00,
                        6.3237e-02,
                        -6.0882e-01,
                        -6.6295e-01,
                        -1.5366e-01,
                        4.9685e-01,
                        6.2274e-01,
                        -1.3168e-01,
                    ],
                    [
                        6.8944e-01,
                        -3.9142e-01,
                        -7.1973e-01,
                        4.3688e-01,
                        -3.6193e-02,
                        6.2854e-02,
                        -1.1850e+00,
                        5.0581e-02,
                    ],
                ],
            ],
            [
                [
                    [
                        -1.6655e-01,
                        8.4826e-02,
                        1.7989e+00,
                        -7.8037e-01,
                        1.3212e+00,
                        -1.2101e+00,
                        -2.2851e-01,
                        1.7132e-01,
                    ],
                    [
                        -5.0339e-03,
                        1.4641e-01,
                        -5.2717e-01,
                        1.3210e+00,
                        7.7905e-02,
                        4.3598e-01,
                        9.3168e-01,
                        3.1351e+00,
                    ],
                    [
                        3.3921e-01,
                        -8.1749e-01,
                        8.1084e-01,
                        2.7485e-02,
                        1.6608e+00,
                        6.9049e-01,
                        -9.6010e-01,
                        -5.5352e-03,
                    ],
                    [
                        3.3315e-02,
                        9.5610e-01,
                        1.7875e-01,
                        -4.4073e-01,
                        -5.4367e-01,
                        1.7259e+00,
                        2.2236e+00,
                        -2.6123e+00,
                    ],
                ],
                [
                    [
                        5.8184e-01,
                        1.2369e+00,
                        -3.9746e-01,
                        6.5081e-01,
                        -2.3186e-01,
                        -1.1190e+00,
                        7.0249e-01,
                        -4.7945e-01,
                    ],
                    [
                        1.1524e+00,
                        1.9687e+00,
                        1.7438e-01,
                        -1.1724e+00,
                        -1.1427e+00,
                        1.0857e+00,
                        1.5862e-01,
                        -4.9271e-01,
                    ],
                    [
                        2.1777e+00,
                        3.9714e-01,
                        7.1685e-01,
                        -1.3168e+00,
                        -1.4173e+00,
                        3.4189e-01,
                        8.0982e-01,
                        2.3465e-01,
                    ],
                    [
                        3.5936e-01,
                        1.3531e+00,
                        -6.1714e-01,
                        -3.1898e-01,
                        1.6157e+00,
                        -7.4338e-01,
                        -4.3354e-01,
                        -3.3896e-01,
                    ],
                ],
                [
                    [
                        1.4062e+00,
                        6.2594e-01,
                        1.9250e+00,
                        1.3963e-01,
                        1.7675e+00,
                        -4.8651e-01,
                        9.9848e-01,
                        -1.0034e+00,
                    ],
                    [
                        6.9961e-01,
                        2.5290e-01,
                        7.5512e-01,
                        7.3720e-01,
                        -9.0817e-01,
                        -1.7826e+00,
                        5.4702e-01,
                        -4.2474e-01,
                    ],
                    [
                        3.2706e-01,
                        -4.9742e-01,
                        -1.2587e+00,
                        -1.2843e-01,
                        -1.2756e-01,
                        -5.3438e-02,
                        4.6741e-01,
                        -1.5908e+00,
                    ],
                    [
                        -4.7023e-01,
                        2.9533e-02,
                        4.1307e-01,
                        -1.6354e+00,
                        -6.8290e-02,
                        -1.0719e+00,
                        1.0981e+00,
                        1.5012e-02,
                    ],
                ],
            ],
            [
                [
                    [
                        -1.3865e-01,
                        -1.6713e+00,
                        2.4251e-01,
                        -1.4237e+00,
                        1.4950e+00,
                        6.6317e-01,
                        3.9568e-01,
                        -6.8158e-01,
                    ],
                    [
                        -4.0025e-02,
                        -4.9230e-01,
                        7.8022e-01,
                        -1.1951e-01,
                        -9.2484e-01,
                        -8.3802e-01,
                        -1.0279e+00,
                        4.1673e-01,
                    ],
                    [
                        -6.0054e-01,
                        -6.2291e-01,
                        -1.6158e+00,
                        -4.9459e-01,
                        1.4529e+00,
                        -3.5869e-01,
                        -2.5390e+00,
                        -1.3678e+00,
                    ],
                    [
                        2.4040e+00,
                        5.5869e-03,
                        -3.9134e-01,
                        6.6219e-01,
                        -4.6620e-01,
                        3.3898e-01,
                        4.3022e-01,
                        2.2850e-01,
                    ],
                ],
                [
                    [
                        -9.4525e-02,
                        2.1883e+00,
                        -6.7008e-01,
                        -7.2547e-01,
                        6.0386e-01,
                        -9.3348e-02,
                        6.7104e-01,
                        1.1074e+00,
                    ],
                    [
                        -3.9714e-01,
                        -3.6825e-01,
                        -2.2681e+00,
                        2.5561e-01,
                        7.4940e-01,
                        -3.2900e-01,
                        -9.9009e-02,
                        3.6605e-01,
                    ],
                    [
                        -1.3174e+00,
                        1.1532e+00,
                        1.7893e-02,
                        4.4546e-01,
                        1.1974e-01,
                        5.5479e-02,
                        1.0754e-01,
                        5.9695e-01,
                    ],
                    [
                        2.3358e-01,
                        -2.0521e+00,
                        -3.5099e-01,
                        -1.2203e+00,
                        1.2874e+00,
                        2.1670e-01,
                        9.3595e-01,
                        2.3518e-01,
                    ],
                ],
                [
                    [
                        -2.3895e-01,
                        -2.1933e-01,
                        4.3462e-01,
                        -1.6698e+00,
                        -1.2020e+00,
                        1.5044e+00,
                        1.1940e-01,
                        4.7909e-01,
                    ],
                    [
                        -2.0673e-01,
                        2.4016e-01,
                        -2.4454e-01,
                        2.0202e+00,
                        1.3364e+00,
                        -3.5011e-01,
                        2.7738e+00,
                        1.0000e+00,
                    ],
                    [
                        3.2979e-01,
                        -9.7155e-01,
                        4.3309e-01,
                        1.4969e+00,
                        4.2577e-01,
                        1.0230e-01,
                        -3.2127e-01,
                        7.6607e-01,
                    ],
                    [
                        -6.5149e-02,
                        -5.7939e-01,
                        -9.5800e-01,
                        8.5078e-01,
                        -7.8531e-02,
                        1.8368e+00,
                        -6.0313e-02,
                        1.7064e-03,
                    ],
                ],
            ],
        ])
        .to_device(&mut ctx);

        let y_gt = Tensor::new([[
            [
                [
                    -0.9950, 3.8086, 1.0965, -2.0639, 0.8647, -0.2173, -4.0601, -0.9471,
                ],
                [
                    -1.3368, 5.6905, -0.8755, 1.9894, -7.4201, 2.8247, 5.4984, 2.5467,
                ],
                [
                    -9.0210, -6.7905, -6.5291, -2.7378, -1.8391, 3.8629, -8.0257, 7.3519,
                ],
                [
                    -3.7796, 0.8218, 5.1904, -4.3419, -5.8769, -4.9956, 4.1278, 5.1438,
                ],
                [
                    1.0262, 5.9446, -4.9197, -1.7199, 0.7524, 3.6105, -2.7893, -8.7710,
                ],
            ],
            [
                [
                    -3.7209, 1.8734, -1.6744, -2.5943, 4.2209, 4.5146, -1.3314, -5.0417,
                ],
                [
                    -0.7865, -12.2317, 5.5228, 16.1922, 2.0082, 3.0134, 1.1522, -3.8370,
                ],
                [
                    1.7378, -4.9991, -5.6781, 5.4619, 8.8425, -0.6570, 2.0100, -4.6027,
                ],
                [
                    13.6469, -0.0445, -3.8885, 7.0379, -9.5465, 0.6857, -8.3285, -1.0291,
                ],
                [
                    3.1371, -3.3014, 5.7100, -0.7889, -5.0901, -6.1664, 1.0991, -5.9197,
                ],
            ],
            [
                [
                    -5.7700, 4.6635, -1.7656, 0.2473, -0.7631, -4.8032, -2.0550, 4.1603,
                ],
                [
                    -13.8981, -4.6691, 4.0304, -2.9285, -3.7400, -0.9466, 1.3042, 0.2157,
                ],
                [
                    1.3434, 5.8707, -5.7859, -10.3137, -3.0729, 1.0717, 0.0475, -6.0326,
                ],
                [
                    -2.4243, 5.0810, -0.7792, -2.8177, -5.8776, 5.6378, 3.1555, -1.4040,
                ],
                [
                    0.2051, -5.6506, 1.5098, 4.0439, 1.0034, 6.4480, -0.5522, -1.0986,
                ],
            ],
            [
                [
                    -5.3252, -5.3361, 0.7436, 4.2839, -0.2710, 3.0569, 0.4509, 3.0842,
                ],
                [
                    4.3720, 0.8407, -3.7872, 1.8441, -1.2920, 6.7858, -19.1767, 1.2731,
                ],
                [
                    -0.9500, -3.8676, 0.9383, -9.6816, 0.9959, -0.9475, -4.3760, -8.9343,
                ],
                [
                    -0.7683, 3.4058, 4.6833, 6.5988, -3.1161, -0.0220, -10.0812, -1.6278,
                ],
                [
                    -8.6986, 0.5455, 6.2556, 3.1038, 2.7789, 1.9541, -4.0711, -5.8953,
                ],
            ],
            [
                [
                    8.4716, 1.8929, 4.9629, -0.4739, 4.2110, 0.3370, 3.4055, -5.5942,
                ],
                [
                    2.1359, 4.1030, -3.3563, -4.2121, -0.2962, 4.5888, 3.7409, -3.9385,
                ],
                [
                    2.4848, -3.1268, 3.5773, 0.4729, 6.5174, 1.4893, -3.8606, -0.5915,
                ],
                [
                    -2.4115, -5.8014, 0.6162, 6.8059, -1.0850, -4.7522, -2.1020, -1.5636,
                ],
                [
                    0.9375, -2.8156, 3.4800, 1.0592, -2.7793, -0.2448, -5.6488, 4.2475,
                ],
            ],
        ]]);

        let x = Var::new(x);
        let filter = Var::new(filter);

        let y = conv_2d(&x, &filter, [1, 1], [1, 1], [1, 1]);

        assert!(Tensor::all_close(&y.eval(&mut ctx), &y_gt, 0.001));

        // conv grad
        assert!(grad_check(&y, &filter, 0.1, &mut ctx));
        assert!(grad_check(&y, &x, 0.1, &mut ctx));
    }

    #[test]
    pub fn test_conv_transpose_2d() {}
}
