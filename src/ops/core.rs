use crate::error::Error;
use crate::ops;
use crate::ops::{Category, NaryOperator, VariadicOperator};
use crate::shape::{display_comma, Array, Axes, Axis, Extent, Shape};
use crate::tensor::data::{DataLiteral, DataType};
use crate::var::{Var, Variable};
use itertools::Itertools;
use std::time::Instant;

use crate::ops::map::{scalar, MapOperator, NullaryMap, NullaryMapOperator, UnaryMapOperator};
use crate::session::context::Context;
use crate::tensor::{Tensor, TensorDesc};

#[derive(Clone, Debug)]
pub enum LayoutOperator {
    Expand(Expand),
}

// copy
#[derive(Clone, Debug)]
struct Reshape {
    input: [TensorDesc; 1],
    output: TensorDesc,
}

// no-copy
#[derive(Clone, Debug)]
pub struct Expand {
    input: [TensorDesc; 1],
    output: TensorDesc,
}

#[derive(Clone, Debug)]
struct Permute {
    input: [TensorDesc; 1],
    output: TensorDesc,
    perm: Array,
}

#[derive(Clone, Debug)]
struct Squeeze {
    input: [TensorDesc; 1],
    output: TensorDesc,
    axis: usize,
}

#[derive(Clone, Debug)]
struct Unsqueeze {
    input: [TensorDesc; 1],
    output: TensorDesc,
    axis: usize,
}

#[derive(Clone, Debug)]
struct Slice {
    input: [TensorDesc; 1],
    output: TensorDesc,
    axis: usize,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct Unslice {
    input: [TensorDesc; 1],
    output: TensorDesc,
    axis: usize,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct Gather {
    input: [TensorDesc; 2],
    output: TensorDesc,
    axis: usize,
}

#[derive(Clone, Debug)]
struct Scatter {
    input: [TensorDesc; 2],
    output: TensorDesc,
    axis: usize,
}

#[derive(Clone, Debug)]
struct Concat {
    input: Vec<TensorDesc>,
    output: TensorDesc,
    axis: usize,
}

pub fn ceil_div(a: usize, b: usize) -> usize {
    1 + ((a - 1) / b)
}

pub fn ceil(a: usize, b: usize) -> usize {
    ceil_div(a, b) * b
}

pub fn div_up(a: usize, b: usize) -> usize {
    (a + (b - 1)) / b
}

pub fn inc_arr<const N: usize>(start: usize) -> [usize; N] {
    let mut x = [0; N];
    for i in 0..N {
        x[i] = i + start;
    }
    x
}

pub fn translate_id<S1, S2>(
    r_id: S1,
    r_stride: &[usize],
    t_ids: &[S2],
    t_strides: &[&[usize]],
    t_offsets: &[usize],
) -> (String, bool)
where
    S1: AsRef<str>,
    S2: AsRef<str>,
{
    let ndim = r_stride.len();

    assert_eq!(t_ids.len(), t_strides.len());
    assert_eq!(t_strides.len(), t_offsets.len());

    let mut def_c = Vec::new();
    let mut strides_c = Vec::new();
    let mut idx_add_c = Vec::new();
    let r_id = r_id.as_ref();
    for ((id, stride), offset) in t_ids.iter().zip(t_strides).zip(t_offsets) {
        // no need for translation
        let id = id.as_ref();
        if r_stride.eq(*stride) {
            def_c.push(format!("uint {id} = {offset} + {r_id};"));
        } else {
            def_c.push(format!("uint {id} = {offset};"));
            strides_c.push(format!(
                "uint t_strides_{id}[{ndim}] = {{{}}};",
                display_comma(stride)
            ));
            idx_add_c.push(format!("{id} += c * t_strides_{id}[i];"));
        }
    }

    let def_s = def_c.join("\n");

    if !idx_add_c.is_empty() {
        strides_c.push(format!(
            "uint r_strides[{ndim}] = {{{}}};",
            display_comma(r_stride)
        ));
        let stride_s = strides_c.join("\n");
        let idx_add_s = idx_add_c.join("\n");
        (
            format!(
                r#"
            {def_s}
            {stride_s}
            uint rem = {r_id};
            for (uint i = 0; i < {ndim}; ++i) {{
                uint c = rem / r_strides[i];
                rem -= c * r_strides[i];
                {idx_add_s}
            }}
        "#
            ),
            false,
        )
    } else {
        (def_s, true)
    }
}

////////////// common routines
//
// pub fn init_zero(x: &Tensor, ctx: &mut Context) {
//     let data_type = x.data_type().opencl();
//
//     let k = ctx
//         .get_program(format!(
//             r#"
//         __kernel void zero(__global {data_type} *x) {{
//             uint gid = get_global_id(0);
//             x[gid] = 0;
//         }}"#
//         ))
//         .kernel("zero")
//         .arg_tensor(x)
//         .global_work_size(x.size());
//
//     //let now = Instant::now();
//
//     k.launch();
//
//     //println!(" elapsed: {:.2?} / {}", now.elapsed(), x.size());
// }

////////////// ---------------------------------------------------

pub fn view<V, E>(x: V, extents: E) -> Var
where
    V: Variable,
    E: Extent,
{
    let x = x.into_var();
    let extents = extents.to_arr(x.size()).unwrap();
    reshape(x, Shape::new(&extents))
}

pub fn reshape<V>(x: V, shape: Shape) -> Var
where
    V: Variable,
{
    let x = x.into_var().organize();

    if x.size() != shape.size() {
        panic!("incompatible size");
    }

    Var::from_unary_op(
        Reshape {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape,
                data_type: x.data_type(),
            },
        },
        x,
    )
}

// always creates new tensor
impl NaryOperator<1> for Reshape {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Var; 1], _: &Var, gy: &Var) -> [Option<Var>; 1] {
        [Some(reshape(gy, self.input[0].shape.clone()))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        let mut shape = self.output.shape.clone();
        shape.offset += x[0].offset();

        Ok(Tensor {
            desc: TensorDesc {
                shape,
                data_type: x[0].data_type(),
            },
            data: x[0].data.clone(),
        })
    }
}

pub fn expand<V, E>(x: V, extents: E) -> Var
where
    V: Variable,
    E: Extent,
{
    let x = x.into_var();
    let extents = extents.to_arr(0).unwrap();

    Var::from_unary_op(
        Expand {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: x.shape().expand(&extents),
                data_type: x.data_type(),
            },
        },
        x,
    )
}

impl NaryOperator<1> for Expand {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Var; 1], _: &Var, gy: &Var) -> [Option<Var>; 1] {
        // match dims from the backwards

        let surplus = gy.rank() - x[0].rank();
        let mut axes = Array::new();

        for i in 0..surplus {
            axes.push(i);
        }

        for i in 0..x[0].rank() {
            if gy.extent(i + surplus) != x[0].extent(i) {
                axes.push(i + surplus);
            }
        }

        // println!("gy {:?}", gy.extents());
        // println!("expected gx {:?}", x.extents());
        // println!("actual gx {:?}", gy.sum(&axes, false).extents());

        [Some(gy.sum(axes, false).view(self.input[0].extents()))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        Ok(x[0].expand(self.output().shape.extents()))
    }

    fn cat(&self) -> Category {
        Category::Layout(LayoutOperator::Expand(self.clone()))
    }
}

pub fn tr<V>(x: V) -> Var
where
    V: Variable,
{
    transpose(x, 0, 1)
}

pub fn transpose<V, A1, A2>(x: V, axis1: A1, axis2: A2) -> Var
where
    V: Variable,
    A1: Axis,
    A2: Axis,
{
    let x = x.into_var();
    let rank = x.rank();

    let axis1 = axis1.to_usize(rank).unwrap();
    let axis2 = axis2.to_usize(rank).unwrap();

    let axes = (0..rank)
        .map(|i| {
            if i == axis1 {
                axis2
            } else if i == axis2 {
                axis1
            } else {
                i
            }
        })
        .collect::<Array>();

    permute(x, axes)
}

pub fn permute<V, A>(x: V, axes: A) -> Var
where
    V: Variable,
    A: Axes,
{
    let x = x.into_var();
    let axes = axes.to_arr(x.rank()).unwrap();

    Var::from_unary_op(
        Permute {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: x.shape().permute(axes.as_slice()),
                data_type: x.data_type(),
            },
            perm: axes,
        },
        x,
    )
}

impl NaryOperator<1> for Permute {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Var; 1], _: &Var, gy: &Var) -> [Option<Var>; 1] {
        // [A, B, C, D] -> permute(2, 1, 4, 3) -> [B, A, D, C]
        // [B, A, D, C] -> permute(2, 1, 4, 3) -> [A, B, C, D]

        [Some(permute(gy, self.perm.as_slice()))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        Ok(x[0].permute(self.perm.as_slice()))
    }
}

pub fn squeeze<V, A>(x: V, axis: A) -> Var
where
    V: Variable,
    A: Axis,
{
    let mut x = x.into_var();
    let axis = axis.to_usize(x.rank()).unwrap();

    Var::from_unary_op(
        Squeeze {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: x.shape().remove(axis),
                data_type: x.data_type(),
            },
            axis,
        },
        x,
    )
}

impl NaryOperator<1> for Squeeze {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Var; 1], _: &Var, gy: &Var) -> [Option<Var>; 1] {
        [Some(unsqueeze(gy, self.axis))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        Ok(x[0].squeeze_axis(self.axis))
    }
}

pub fn unsqueeze<V, A>(x: V, axis: A) -> Var
where
    V: Variable,
    A: Axis,
{
    let x = x.into_var();
    let axis = axis.to_usize(x.rank() + 1).unwrap();

    Var::from_unary_op(
        Unsqueeze {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: x.shape().insert(axis),
                data_type: x.data_type(),
            },
            axis,
        },
        x,
    )
}

impl NaryOperator<1> for Unsqueeze {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Var; 1], _: &Var, gy: &Var) -> [Option<Var>; 1] {
        [Some(squeeze(gy, self.axis))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        Ok(x[0].expand_axis(self.axis))
    }
}

pub fn slice<V, A>(x: V, axis: A, start: usize, end: usize) -> Var
where
    V: Variable,
    A: Axis,
{
    let x = x.into_var();
    let axis = axis.to_usize(x.rank()).unwrap();

    Var::from_unary_op(
        Slice {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: x.shape().select_range(start, end - 1, axis),
                data_type: x.data_type(),
            },
            axis,
            start,
            end,
        },
        x,
    )
}

impl NaryOperator<1> for Slice {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Var; 1], _: &Var, gy: &Var) -> [Option<Var>; 1] {
        // scatter add
        [Some(unslice(
            gy,
            self.axis,
            self.start,
            x[0].extent(self.axis),
        ))]
    }

    fn compute(&self, x: [&Tensor; 1], _: &mut Context) -> Result<Tensor, Error> {
        Ok(x[0].slice(self.start, self.end, self.axis))
    }
}

pub fn unslice<V, A>(x: V, axis: A, start: usize, end: usize) -> Var
where
    V: Variable,
    A: Axis,
{
    let x = x.into_var();
    let axis = axis.to_usize(x.rank()).unwrap();

    assert!(end >= start + x.extent(axis));

    let mut extents = x.extents().to_vec();
    extents[axis] = end;

    Var::from_unary_op(
        Unslice {
            input: [x.desc().clone()],
            output: TensorDesc {
                shape: Shape::new(extents),
                data_type: x.data_type(),
            },
            axis,
            start,
            end,
        },
        x,
    )
}

impl NaryOperator<1> for Unslice {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Var; 1], _: &Var, gy: &Var) -> [Option<Var>; 1] {
        [Some(slice(
            gy,
            self.axis,
            self.start,
            self.start + x[0].extent(self.axis),
        ))]
    }

    fn compute(&self, x: [&Tensor; 1], ctx: &mut Context) -> Result<Tensor, Error> {
        let x = x[0];
        let mut extents = x.extents().to_vec();
        extents[self.axis] = self.end;

        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors
        y.fill_zero();

        let stride_ref = Shape::default_strides(x.extents());

        let data_type = x.data_type().opencl();
        // computer.submit(ctx.backend.unslice())
        let (idx_c, is_direct) = translate_id(
            "gid",
            &stride_ref,
            &["idx_x", "idx_y"],
            &[x.strides(), y.strides()],
            &[x.offset(), y.offset() + self.start * y.strides()[self.axis]],
        );

        let p = ctx.get_program(format!(
            r#"
        __kernel void unslice(
            __global const {data_type} *x,
            __global {data_type} *y) {{
            uint gid = get_global_id(0);
            {idx_c}
            y[idx_y] = x[idx_x];
        }} "#
        ));

        p.kernel("unslice")
            .arg_tensor(x)
            .arg_tensor(&y)
            .global_work_size(x.size())
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }
}

pub fn gather<V1, V2, A>(x: V1, ind: V2, axis: A) -> Var
where
    V1: Variable,
    V2: Variable,
    A: Axis,
{
    let x = x.into_var();
    let ind = ind.into_var();
    let axis = axis.to_usize(x.rank()).unwrap();

    assert_eq!(x.rank(), ind.rank());

    // (164, 100) g0 (64, 100) -> (64, 1)
    assert!((0..x.rank()).any(|i| i == axis || x.extent(i) >= ind.extent(i)));

    Var::from_binary_op(
        Gather {
            input: [x.desc().clone(), ind.desc().clone()],
            output: TensorDesc {
                shape: Shape::new(ind.extents()),
                data_type: x.data_type(),
            },
            axis,
        },
        x,
        ind,
    )
}

impl NaryOperator<2> for Gather {
    fn input(&self) -> &[TensorDesc; 2] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Var; 2], _: &Var, gy: &Var) -> [Option<Var>; 2] {
        [Some(scatter(gy, x[1], self.axis, x[0].extents())), None]
    }

    fn compute(&self, x: [&Tensor; 2], ctx: &mut Context) -> Result<Tensor, Error> {
        let strides = display_comma(x[0].strides());
        let strides_idx = display_comma(x[1].strides());

        let strides_d = display_comma(Shape::default_strides(x[1].extents()).as_slice());
        let ndim = x[0].rank(); // == ind.order();
        let axis = self.axis;
        let data_type = x[0].data_type().opencl();

        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        let (gather_id_c, _) = translate_id(
            "gid",
            &Shape::default_strides(x[1].extents()),
            &["gather_idx"],
            &[x[1].strides()],
            &[0],
        );

        let p = ctx.get_program(format!(
            r#"
            __kernel void gather(
                __global const {data_type} *x,
                __private const int x_offset,
                __global const uint *ind,
                __private const int ind_offset,
                __global {data_type} *y,
                __private const int y_offset) {{

                uint gid = get_global_id(0);
                
                uint idx = x_offset;

                {gather_id_c}
                uint target_idx = ind[gather_idx + ind_offset];

                uint strides[{ndim}] = {{{strides}}};
                uint strides_d[{ndim}] = {{{strides_d}}};
                uint rem2 = gid;
                for (uint i = 0; i < {ndim}; ++i) {{
                    uint c = rem2 / strides_d[i];
                    rem2 -= c * strides_d[i];
                    if (i == {axis} ) {{
                        idx += target_idx * strides[i];
                    }} else {{
                        idx += c * strides[i];
                    }}
                }}
                y[gid + y_offset] = x[idx];
             }} "#
        ));

        p.kernel("gather")
            .arg_tensor(x[0])
            .arg(x[0].offset() as i32)
            .arg_tensor(x[1])
            .arg(x[1].offset() as i32)
            .arg_tensor(&y)
            .arg(y.offset() as i32)
            .global_work_size(y.size())
            .launch()
            .map_err(Error::Device)?;
        Ok(y)
    }
}

pub fn scatter<V1, V2, A, E>(x: V1, ind: V2, axis: A, extents: E) -> Var
where
    V1: Variable,
    V2: Variable,
    A: Axis,
    E: Extent,
{
    let x = x.into_var();
    let ind = ind.into_var();
    let axis = axis.to_usize(x.rank()).unwrap();
    let extents = extents.to_arr(0).unwrap();

    assert_eq!(extents.len(), x.rank());
    assert_eq!(x.rank(), ind.rank());

    // (64, 1) -> (64, 1) with (64, 10)
    assert!((0..x.rank())
        .any(|i| { i == axis || (x.extent(i) >= ind.extent(i) && extents[i] >= x.extent(i)) }));

    Var::from_binary_op(
        Scatter {
            input: [x.desc().clone(), ind.desc().clone()],
            output: TensorDesc::new(extents, x.data_type()),
            axis,
        },
        x,
        ind,
    )
}

impl NaryOperator<2> for Scatter {
    fn input(&self) -> &[TensorDesc; 2] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Var; 2], _: &Var, gy: &Var) -> [Option<Var>; 2] {
        [Some(gather(gy, x[1], self.axis)), None]
    }

    fn compute(&self, x: [&Tensor; 2], ctx: &mut Context) -> Result<Tensor, Error> {
        let data_type = x[0].data_type();
        let strides_src = display_comma(x[0].strides());
        let strides_dst = display_comma(self.output.shape().strides());
        let strides_d = display_comma(x[1].strides());

        let ndim = self.output.shape().num_axes();
        let axis = self.axis;

        let (scatter_id_c, _) = translate_id(
            "gid",
            &Shape::default_strides(x[1].extents()),
            &["scatter_idx"],
            &[x[1].strides()],
            &[0],
        );

        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors
        y.fill_zero();

        let p = ctx.get_program(format!(
            r#"
        __kernel void scatter(__global const {data_type} *x,
            __private const int x_offset,
            __global const uint *ind,
            __private const int ind_offset,
            __global {data_type} *y,
            __private const int y_offset) {{

            uint gid = get_global_id(0);
            uint idx_in = x_offset;
            uint idx_out = y_offset;

            uint strides_dst[{ndim}] = {{{strides_dst}}};
            uint strides_src[{ndim}] = {{{strides_src}}};
            uint strides_d[{ndim}] = {{{strides_d}}};

            {scatter_id_c}

            uint target_idx = ind[scatter_idx + ind_offset];

            uint rem2 = gid;
            for (uint i = 0; i < {ndim}; ++i) {{
                uint c = rem2 / strides_d[i];
                rem2 -= c * strides_d[i];
                idx_in += c * strides_src[i];
                if (i == {axis}) {{
                    idx_out += target_idx * strides_dst[i];
                }} else {{
                    idx_out += c * strides_dst[i];
                }}
            }}
            y[idx_out + y_offset] += x[idx_in];
        }}
        "#
        ));

        p.kernel("scatter")
            .arg_tensor(x[0])
            .arg(x[0].offset() as i32)
            .arg_tensor(x[1])
            .arg(x[1].offset() as i32)
            .arg_tensor(&y)
            .arg(y.offset() as i32)
            .global_work_size(x[1].size())
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }
}

pub fn concat<V, I, A>(x: I, axis: A) -> Var
where
    V: Variable,
    I: IntoIterator<Item = V>,
    A: Axis,
{
    let x: Vec<Var> = x.into_iter().map(|v| v.into_var()).collect_vec();
    let axis = axis.to_usize(x[0].rank()).unwrap();

    let mut concat_size = x[0].extent(axis);

    for v in x.iter().skip(1) {
        if v.rank() != x[0].rank()
            || v.extents()[..axis] != x[0].extents()[..axis]
            || v.extents()[(axis + 1)..] != x[0].extents()[(axis + 1)..]
        {
            panic!("not in the same shape");
        }
        concat_size += v.extent(axis);
    }

    let mut extents = x[0].extents().to_vec();
    extents[axis] = concat_size;

    Var::from_variadic_op(
        Concat {
            input: x.iter().map(|v| v.desc()).cloned().collect_vec(),
            output: TensorDesc {
                shape: Shape::new(extents),
                data_type: x[0].data_type(),
            },
            axis,
        },
        x,
    )
}

impl VariadicOperator for Concat {
    fn input(&self) -> &[TensorDesc] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: &[Var], _: &Var, gy: &Var) -> Vec<Option<Var>> {
        let mut i = 0;
        let mut g = Vec::new();
        for e in x.iter().map(|v| v.extents()[self.axis]) {
            g.push(Some(slice(gy, self.axis, i, i + e)));
            i += e;
        }
        g
    }

    fn compute(&self, x: &[Tensor], ctx: &mut Context) -> Result<Tensor, Error> {
        let data_type = x[0].data_type();

        //let y = Tensor::zeros(self.output().shape().extents()).to_device(ctx); // all defaults to gpu tensors
        let y = Tensor::uninit2(self.output(), ctx)?;
        y.fill_zero();

        let mut i = 0;

        for v in x {
            let e = v.extent(self.axis);

            let y_sub_shape = y.shape().select_range(i, i + e - 1, self.axis);
            i += e;

            let stride_ref = Shape::default_strides(v.extents());

            let (idx_c, is_direct) = translate_id(
                "gid",
                &stride_ref,
                &["idx_in", "idx_out"],
                &[v.strides(), y_sub_shape.strides()],
                &[v.offset(), y_sub_shape.offset()],
            );

            let p = ctx.get_program(format!(
                r#"
                __kernel void concat(__global const {data_type} *x, __global {data_type} *y) {{
                    uint gid = get_global_id(0);
                    {idx_c}
                    y[idx_out] = x[idx_in];
                }}"#
            ));

            p.kernel("concat")
                .arg_tensor(v)
                .arg_tensor(&y)
                .global_work_size(v.size())
                .launch()
                .map_err(Error::Device)?;
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::core::{concat, expand, gather, permute, slice, squeeze, unslice, unsqueeze};
    use crate::session::context::Context;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use crate::var::{grad_check, Var};

    #[test]
    fn test_reshape() {
        let mut ctx = Context::new();

        let a = Tensor::new([
            [
                [0.5224, -1.2505, 1.6338, 0.2695],
                [1.0670, 0.4321, -2.0123, 0.5689],
            ],
            [
                [0.5790, 0.4326, 0.1468, -1.3014],
                [0.3314, 1.0172, 0.8646, -0.3811],
            ],
            [
                [0.5414, -0.6000, 1.1408, 2.0662],
                [-0.8066, -1.2772, -1.5542, 0.2278],
            ],
        ])
        .to_device(&mut ctx);

        // contiguous case
        let b_gt = Tensor::new([
            [[0.5224, -1.2505, 1.6338], [0.2695, 1.0670, 0.4321]],
            [[-2.0123, 0.5689, 0.5790], [0.4326, 0.1468, -1.3014]],
            [[0.3314, 1.0172, 0.8646], [-0.3811, 0.5414, -0.6000]],
            [[1.1408, 2.0662, -0.8066], [-1.2772, -1.5542, 0.2278]],
        ]);

        let c_gt = Tensor::new([
            [[0.5224, -1.2505, 1.6338], [-2.0123, 0.5689, 0.5790]],
            [[0.3314, 1.0172, 0.8646], [1.1408, 2.0662, -0.8066]],
            [[0.2695, 1.0670, 0.4321], [0.4326, 0.1468, -1.3014]],
            [[-0.3811, 0.5414, -0.6000], [-1.2772, -1.5542, 0.2278]],
        ]);

        let a = Var::new(a);

        let b = a.reshape(Shape::new([4, 2, 3]));

        let c = b.transpose(0, 1).reshape(Shape::new([4, 2, 3]));

        assert!(Tensor::all_close(&b.eval(&mut ctx), &b_gt, 0.001));
        assert!(grad_check(&b, &a, 0.01, &mut ctx));

        assert!(Tensor::all_close(&c.eval(&mut ctx), &c_gt, 0.001));
        assert!(grad_check(&c, &a, 0.01, &mut ctx));
    }

    #[test]
    fn test_expand() {
        let mut ctx = Context::new();

        let x = Tensor::new([[1.0], [2.0], [3.0]]);
        let y_gt = Tensor::new([
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]);

        let y = expand(x, [3, 4]).eval(&mut ctx);

        assert!(Tensor::all_close(&y, &y_gt, 0.001));
    }

    #[test]
    fn test_permute() {
        let mut ctx = Context::new();

        let x = Tensor::new([
            [
                [
                    [[-0.7600], [0.9701], [-0.0958]],
                    [[0.1336], [1.7030], [-0.7256]],
                ],
                [
                    [[-0.6641], [1.1803], [-0.4685]],
                    [[-0.0786], [1.3310], [1.8260]],
                ],
            ],
            [
                [
                    [[0.0339], [-0.7861], [1.5770]],
                    [[-0.8663], [-0.8559], [1.2322]],
                ],
                [
                    [[0.3555], [2.1675], [-0.2272]],
                    [[0.1420], [-1.6608], [-1.6775]],
                ],
            ],
            [
                [
                    [[-1.0854], [0.2658], [-1.0028]],
                    [[-0.2861], [0.5694], [0.4414]],
                ],
                [
                    [[-0.3928], [0.0731], [-0.7040]],
                    [[-0.4140], [-0.1104], [-0.1217]],
                ],
            ],
        ]);

        let y_gt = Tensor::new([
            [
                [[[-0.7600, 0.0339, -1.0854], [-0.6641, 0.3555, -0.3928]]],
                [[[0.9701, -0.7861, 0.2658], [1.1803, 2.1675, 0.0731]]],
                [[[-0.0958, 1.5770, -1.0028], [-0.4685, -0.2272, -0.7040]]],
            ],
            [
                [[[0.1336, -0.8663, -0.2861], [-0.0786, 0.1420, -0.4140]]],
                [[[1.7030, -0.8559, 0.5694], [1.3310, -1.6608, -0.1104]]],
                [[[-0.7256, 1.2322, 0.4414], [1.8260, -1.6775, -0.1217]]],
            ],
        ]);

        let y = permute(x, [2, 3, 4, 1, 0]).eval(&mut ctx);

        assert!(Tensor::all_close(&y, &y_gt, 0.001));
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let x = Tensor::zeros([2, 3, 1, 1, 3]);

        assert_eq!(squeeze(&x, 2).extents(), &[2, 3, 1, 3]);
        assert_eq!(squeeze(&x, 3).extents(), &[2, 3, 1, 3]);

        assert_eq!(unsqueeze(&x, 2).extents(), &[2, 3, 1, 1, 1, 3]);
        assert_eq!(unsqueeze(&x, 3).extents(), &[2, 3, 1, 1, 1, 3]);
    }

    #[test]
    fn test_slice() {
        let mut ctx = Context::new();

        let x = Tensor::new([
            [
                [-0.5671, -0.6673, -0.2867],
                [-0.3271, 0.1534, 1.5160],
                [0.4896, -1.1891, -0.9236],
            ],
            [
                [0.4878, 0.0070, -0.1518],
                [1.8266, -0.6355, 3.3751],
                [-0.5205, 0.7084, -1.3345],
            ],
            [
                [-0.5663, -0.3056, -0.7709],
                [0.0530, 0.0954, 0.1056],
                [-1.9441, -2.0164, -0.7940],
            ],
        ]);

        let y_gt = Tensor::new([
            [[-0.3271, 0.1534, 1.5160]],
            [[1.8266, -0.6355, 3.3751]],
            [[0.0530, 0.0954, 0.1056]],
        ]);

        let y = slice(x, 1, 1, 2).eval(&mut ctx);
        assert!(Tensor::all_close(&y, &y_gt, 0.001));
    }

    #[test]
    fn test_unslice() {
        let mut ctx = Context::new();

        let x = Tensor::new([
            [[-1.0409, -1.9208, 0.3739]],
            [[-0.8022, 1.5944, 0.0105]],
            [[0.0488, 0.3315, -0.1204]],
        ])
        .to_device(&mut ctx);

        let y_gt = Tensor::new([
            [
                [0.0000, 0.0000, 0.0000],
                [-1.0409, -1.9208, 0.3739],
                [0.0000, 0.0000, 0.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [-0.8022, 1.5944, 0.0105],
                [0.0000, 0.0000, 0.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.0488, 0.3315, -0.1204],
                [0.0000, 0.0000, 0.0000],
            ],
        ]);
        let y = unslice(x, 1, 1, 3).eval(&mut ctx);

        assert!(Tensor::all_close(&y, &y_gt, 0.001));
    }

    #[test]
    fn test_gather() {
        let mut ctx = Context::new();
        let x = Tensor::new([
            [
                [-0.2270, 1.3071, -0.0051],
                [0.9399, -0.6349, 1.7424],
                [1.4766, 0.6227, 0.5518],
            ],
            [
                [0.2473, 0.0377, -0.6137],
                [0.6982, 1.4135, 1.1811],
                [-1.8595, -0.3687, 0.7054],
            ],
            [
                [-1.3484, -1.4077, 0.7453],
                [-1.0535, 0.3663, -0.4827],
                [-0.4859, 1.4991, 1.3875],
            ],
        ])
        .to_device(&mut ctx);

        let t = Tensor::new([
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ])
        .to_device(&mut ctx);

        let y_gt = Tensor::new([
            [
                [1.3071, 1.3071, 1.3071],
                [-0.6349, -0.6349, -0.6349],
                [0.6227, 0.6227, 0.6227],
            ],
            [
                [0.0377, 0.0377, 0.0377],
                [1.4135, 1.4135, 1.4135],
                [-0.3687, -0.3687, -0.3687],
            ],
            [
                [-1.4077, -1.4077, -1.4077],
                [0.3663, 0.3663, 0.3663],
                [1.4991, 1.4991, 1.4991],
            ],
        ]);

        let y = gather(x, t, 2).eval(&mut ctx);

        assert!(Tensor::all_close(&y, &y_gt, 0.001));
    }

    #[test]
    fn test_scatter() {}

    #[test]
    fn test_concat() {
        let mut ctx = Context::new();

        let x1 = Tensor::new([
            [[-1.6501, 0.1539], [0.4262, -0.8447]],
            [[-0.7969, -0.1363], [-0.1136, -0.4466]],
        ])
        .to_device(&mut ctx);

        let x2 = Tensor::new([[[0.2334, -0.6969]], [[0.4371, -1.5120]]]).to_device(&mut ctx);

        let x3 = Tensor::new([
            [[-0.3994, 0.1328], [-0.1122, -0.0444]],
            [[-0.0357, 0.2009], [0.2449, 1.5443]],
        ])
        .to_device(&mut ctx);

        let y_gt = Tensor::new([
            [
                [-1.6501, 0.1539],
                [0.4262, -0.8447],
                [0.2334, -0.6969],
                [-0.3994, 0.1328],
                [-0.1122, -0.0444],
            ],
            [
                [-0.7969, -0.1363],
                [-0.1136, -0.4466],
                [0.4371, -1.5120],
                [-0.0357, 0.2009],
                [0.2449, 1.5443],
            ],
        ]);

        let y = concat([x1, x2, x3], 1).eval(&mut ctx);

        println!("{:?}", Tensor::all_close(&y, &y_gt, 0.001));
    }
}
