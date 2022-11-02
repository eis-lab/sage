use crate::error::Error;
use crate::ops::core::translate_id;
use crate::ops::{Category, NaryOperator, Operator, TensorDesc, VariadicOperator};
use crate::session::context::{CachedAccess, Context};
use crate::shape;
use crate::shape::{Array, Extent, Shape};
use crate::tensor::data::{DataType, Scalar};
use crate::tensor::Tensor;
use crate::var::{Var, Variable};
use itertools::Itertools;
use core::cell::RefCell;
use core::f32::consts::PI;
use std::fmt::{Debug, Formatter};
use std::ops;
use std::time::Instant;

#[derive(Clone, Debug)]
pub enum MapOperator {
    Nullary(NullaryMapOperator),
    Unary(UnaryMapOperator),
    Binary(BinaryMapOperator),
    Ternary(TernaryMapOperator),
    Variadic(VariadicMapOperator),
}

impl MapOperator {
    pub fn input(&self) -> &[TensorDesc] {
        match self {
            MapOperator::Nullary(opr) => opr.input(),
            MapOperator::Unary(opr) => opr.input(),
            MapOperator::Binary(opr) => opr.input(),
            MapOperator::Ternary(opr) => opr.input(),
            MapOperator::Variadic(opr) => opr.input(),
        }
    }

    pub fn output(&self) -> &TensorDesc {
        match self {
            MapOperator::Nullary(opr) => opr.output(),
            MapOperator::Unary(opr) => opr.output(),
            MapOperator::Binary(opr) => opr.output(),
            MapOperator::Ternary(opr) => opr.output(),
            MapOperator::Variadic(opr) => opr.output(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NullaryMapOperator {
    pub output: TensorDesc,
    pub map: NullaryMap,
    cached: CachedAccess,
}

impl NullaryMapOperator {
    pub fn new(output: TensorDesc, map: NullaryMap) -> Self {
        NullaryMapOperator {
            output,
            map,
            cached: CachedAccess::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum NullaryMap {
    Literal(String),
    //RandomNormal,
    //RandomUniform,
    // Eye, Multinomial, OneHot, ...
}

#[derive(Clone, Debug)]
pub struct UnaryMapOperator {
    pub input: [TensorDesc; 1],
    pub output: TensorDesc,
    pub map: UnaryMap,
    cached: CachedAccess,
}

impl UnaryMapOperator {
    pub fn new(input: TensorDesc, output: TensorDesc, map: UnaryMap) -> Self {
        UnaryMapOperator {
            input: [input],
            output,
            map,
            cached: CachedAccess::new(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum UnaryMap {
    // Used for copying
    Copy,

    Abs,
    Neg,
    Recip,
    Log,
    Exp,
    Sqrt,
    Square,
    Erf,
    Sign,

    Ceil,
    Floor,
    Round,
    // trig
    Sin,
    Sinh,
    Cos,
    Cosh,
    Tan,
    Tanh,
    Asin,
    Asinh,
    Acos,
    Acosh,
    Atan,
    Atanh,

    // Logic
    Not,

    // Cast
    CastInt,
    CastFloat,
}

#[derive(Clone, Debug)]
pub struct BinaryMapOperator {
    pub input: [TensorDesc; 2],
    pub output: TensorDesc,
    pub map: BinaryMap,
    cached: CachedAccess,
}

impl BinaryMapOperator {
    pub fn new(input: [TensorDesc; 2], output: TensorDesc, map: BinaryMap) -> Self {
        BinaryMapOperator {
            input,
            output,
            map,
            cached: CachedAccess::new(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum BinaryMap {
    Add,
    Sub,
    Div,
    Mul,

    Mod,
    Pow,
    Min,
    Max,

    // Logic
    And,
    Or,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

#[derive(Clone, Debug)]
pub struct TernaryMapOperator {
    pub input: [TensorDesc; 3],
    pub output: TensorDesc,
    pub map: TernaryMap,
    cached: CachedAccess,
}

impl TernaryMapOperator {
    pub fn new(input: [TensorDesc; 3], output: TensorDesc, map: TernaryMap) -> Self {
        TernaryMapOperator {
            input,
            output,
            map,
            cached: CachedAccess::new(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum TernaryMap {
    Cond,
}

#[derive(Clone, Debug)]
pub struct VariadicMapOperator {
    pub input: Vec<TensorDesc>,
    pub output: TensorDesc,
    pub expr: Vec<StackElement>,
    cached: CachedAccess,
}

impl VariadicMapOperator {
    pub fn new(input: Vec<TensorDesc>, output: TensorDesc, expr: Vec<StackElement>) -> Self {
        VariadicMapOperator {
            input,
            output,
            expr,
            cached: CachedAccess::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum StackElement {
    Operator(MapOperator),
    Input(usize),
    Operation(String),
}

impl UnaryMapOperator {
    fn map_fn<T>(&self, x: T) -> String
        where
            T: AsRef<str>,
    {
        let x = x.as_ref();
        match self.map {
            UnaryMap::Copy => x.to_string(),
            UnaryMap::Abs => format!("fabs({x})"),
            UnaryMap::Neg => format!("-{x}"),
            UnaryMap::Recip => format!("1/{x}"),
            UnaryMap::Log => format!("log({x})"),
            UnaryMap::Exp => format!("exp({x})"),
            UnaryMap::Sqrt => format!("sqrt({x})"),
            UnaryMap::Square => format!("pow({x}, 2)"),
            UnaryMap::Erf => format!("erf({x})"),
            UnaryMap::Sign => format!("sign({x})"),
            UnaryMap::Ceil => format!("ceil({x})"),
            UnaryMap::Floor => format!("floor({x})"),
            UnaryMap::Round => format!("round({x})"),
            UnaryMap::Sin => format!("sin({x})"),
            UnaryMap::Sinh => format!("sinh({x})"),
            UnaryMap::Cos => format!("cos({x})"),
            UnaryMap::Cosh => format!("cosh({x})"),
            UnaryMap::Tan => format!("tan({x})"),
            UnaryMap::Tanh => format!("tanh({x})"),
            UnaryMap::Asin => format!("asin({x})"),
            UnaryMap::Asinh => format!("asinh({x})"),
            UnaryMap::Acos => format!("acos({x})"),
            UnaryMap::Acosh => format!("acosh({x})"),
            UnaryMap::Atan => format!("atan({x})"),
            UnaryMap::Atanh => format!("atanh({x})"),
            UnaryMap::Not => format!("!{x}"),
            UnaryMap::CastInt => format!("(int){x}"),
            UnaryMap::CastFloat => format!("(float){x}"),
        }
    }
}

impl BinaryMapOperator {
    fn map_fn<T>(&self, x1: T, x2: T) -> String
        where
            T: AsRef<str>,
    {
        let (x1, x2) = (x1.as_ref(), x2.as_ref());

        match self.map {
            BinaryMap::Add => format!("{x1} + {x2}"),
            BinaryMap::Sub => format!("{x1} - {x2}"),
            BinaryMap::Div => format!("{x1} / {x2}"),
            BinaryMap::Mul => format!("{x1} * {x2}"),
            BinaryMap::Mod => format!("mod({x1}, {x2})"),
            BinaryMap::Pow => format!("pow({x1}, {x2})"),
            BinaryMap::Min => format!("min({x1}, {x2})"),
            BinaryMap::Max => format!("max({x1}, {x2})"),
            BinaryMap::And => format!("{x1} && {x2}"),
            BinaryMap::Or => format!("{x1} || {x2}"),
            BinaryMap::Eq => format!("{x1} == {x2}"),
            BinaryMap::Ne => format!("{x1} != {x2}"),
            BinaryMap::Gt => format!("{x1} > {x2}"),
            BinaryMap::Ge => format!("{x1} >= {x2}"),
            BinaryMap::Lt => format!("{x1} < {x2}"),
            BinaryMap::Le => format!("{x1} <= {x2}"),
        }
    }
}

impl TernaryMapOperator {
    fn map_fn<T>(&self, x1: T, x2: T, x3: T) -> String
        where
            T: AsRef<str>,
    {
        let (x1, x2, x3) = (x1.as_ref(), x2.as_ref(), x3.as_ref());

        match self.map {
            TernaryMap::Cond => format!("{x1} ? {x2} : {x3}"),
        }
    }
}

impl VariadicMapOperator {
    fn map_fn<T>(&self, x: &[T]) -> String
        where
            T: AsRef<str>,
    {
        //println!("{:?}", &self.expr);

        let mut expr = self.expr.clone();
        let mut stack = Vec::new();

        //let mut sz = self.sz;
        //let mut offset = start;

        while !expr.is_empty() {
            let e = expr.pop().unwrap();
            match e {
                StackElement::Operator(op) => {
                    let exp = match op {
                        MapOperator::Nullary(op) => match op.map {
                            NullaryMap::Literal(v) => v,
                        },
                        MapOperator::Unary(op) => {
                            let a: String = stack.pop().unwrap();
                            op.map_fn(&a)
                        }
                        MapOperator::Binary(op) => {
                            let a1: String = stack.pop().unwrap();
                            let a2: String = stack.pop().unwrap();
                            op.map_fn(&a1, &a2)
                        }
                        MapOperator::Ternary(op) => {
                            let a1: String = stack.pop().unwrap();
                            let a2: String = stack.pop().unwrap();
                            let a3: String = stack.pop().unwrap();

                            op.map_fn(&a1, &a2, &a3)
                        }
                        MapOperator::Variadic(op) => {
                            let a = stack
                                .drain((stack.len() - op.input().len())..)
                                .rev()
                                .collect_vec();

                            op.map_fn(&a)
                        }
                    };
                    expr.push(StackElement::Operation(format!("({exp})")));
                }
                StackElement::Input(idx) => {
                    stack.push(x[idx].as_ref().to_string());
                }
                StackElement::Operation(exp) => {
                    stack.push(exp);
                }
            }
        }
        assert_eq!(stack.len(), 1);

        let a = stack.pop().unwrap();
        //println!("{:?}", &a);

        a
    }
}

fn compile_map(fn_name: &str, x: &[&Tensor], exp: String, desc: &TensorDesc) -> String {
    let param_c = x
        .iter()
        .enumerate()
        .map(|(i, t)| {
            format!(
                "__global const {dtype} *x{i}_buf",
                dtype = t.data_type().opencl()
            )
        })
        .chain([format!(
            "__global {dtype} *y",
            dtype = desc.data_type.opencl()
        )])
        .join(", ");

    let t_ids = (0..x.len()).map(|i| format!("idx{i}")).collect_vec();
    let t_strides = x.iter().map(|t| t.strides()).collect_vec();
    let t_offsets = x.iter().map(|t| t.offset()).collect_vec();

    let (idx_c, is_direct) = translate_id(
        "gid",
        &Shape::default_strides(desc.extents()),
        &t_ids,
        &t_strides,
        &t_offsets,
    );

    let val_c = (0..x.len())
        .map(|i| format!("{dtype} x{i} = x{i}_buf[idx{i}];", dtype = x[i].data_type()))
        .join("\n");

    format!(
        r#"
        __kernel void {fn_name}({param_c}) {{
            uint gid = get_global_id(0);
            {idx_c}
            {val_c}
            y[gid] = {exp};
        }}"#
    )
}

pub fn broadcast<V1, V2>(x1: V1, x2: V2) -> (Var, Var)
    where
        V1: Variable,
        V2: Variable,
{
    let mut x1 = x1.into_var();
    let mut x2 = x2.into_var();

    let union = shape::union(x1.extents(), x2.extents()).unwrap();

    if x1.extents() != union.as_slice() {
        x1 = x1.expand(&union);
    }
    if x2.extents() != union.as_slice() {
        x2 = x2.expand(&union);
    }
    (x1, x2)
}

pub fn broadcast3<V1, V2, V3>(x1: V1, x2: V2, x3: V3) -> (Var, Var, Var)
    where
        V1: Variable,
        V2: Variable,
        V3: Variable,
{
    let mut x1 = x1.into_var();
    let mut x2 = x2.into_var();
    let mut x3 = x3.into_var();

    let union1 = shape::union(x1.extents(), x2.extents()).unwrap();
    let union2 = shape::union(x2.extents(), x3.extents()).unwrap();
    let union = shape::union(&union1, &union2).unwrap();

    if x1.extents() != union.as_slice() {
        x1 = x1.expand(&union);
    }
    if x2.extents() != union.as_slice() {
        x2 = x2.expand(&union);
    }

    if x3.extents() != union.as_slice() {
        x3 = x3.expand(&union);
    }
    (x1, x2, x3)
}

pub fn unary_map<V>(x: V, map: UnaryMap) -> Var
    where
        V: Variable,
{
    let x = x.into_var();

    let data_type = match map {
        UnaryMap::Sign | UnaryMap::Copy | UnaryMap::Abs | UnaryMap::Neg => x.data_type(),
        UnaryMap::Not => DataType::Uint,
        UnaryMap::CastInt => DataType::Int,
        UnaryMap::CastFloat => DataType::Float,
        _ => DataType::Float,
    };

    Var::from_unary_op(
        UnaryMapOperator {
            input: [x.desc().clone()],
            output: TensorDesc::new(x.extents(), data_type),
            map,
            cached: Default::default(),
        },
        x,
    )
}

pub fn binary_map<V1, V2>(x1: V1, x2: V2, map: BinaryMap) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    let (x1, x2) = broadcast(x1, x2);
    assert_eq!(x1.data_type(), x2.data_type());

    let data_type = match map {
        BinaryMap::And
        | BinaryMap::Or
        | BinaryMap::Eq
        | BinaryMap::Ne
        | BinaryMap::Gt
        | BinaryMap::Ge
        | BinaryMap::Lt
        | BinaryMap::Le => DataType::Uint,
        _ => x1.data_type(),
    };

    Var::from_binary_op(
        BinaryMapOperator {
            input: [x1.desc().clone(), x2.desc().clone()],
            output: TensorDesc::new(x1.extents(), data_type),
            map,
            cached: Default::default(),
        },
        x1,
        x2,
    )
}

///// Nullary ops

pub fn scalar<S, E>(val: S, extents: E) -> Var
    where
        S: Scalar,
        E: Extent,
{
    let data_type = S::data_type();

    let literal = format!("({}){}", data_type.opencl(), val);

    Var::from_nullary_op(NullaryMapOperator {
        output: TensorDesc::new(extents, data_type),
        map: NullaryMap::Literal(literal),
        cached: Default::default(),
    })
}

////// Unary ops

pub fn copy<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Copy)
}

pub fn abs<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Abs)
}

pub fn neg<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Neg)
}

pub fn recip<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Recip)
}

pub fn log<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Log)
}

pub fn exp<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Exp)
}

pub fn sqrt<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Sqrt)
}

pub fn square<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Square)
}

pub fn erf<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Erf)
}

pub fn sign<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Sign)
}

pub fn ceil<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Ceil)
}

pub fn floor<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Floor)
}

pub fn round<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Round)
}

pub fn sin<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Sin)
}

pub fn sinh<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Sinh)
}

pub fn cos<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Cos)
}

pub fn cosh<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Cosh)
}

pub fn tan<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Tan)
}

pub fn tanh<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Tanh)
}

pub fn asin<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Asin)
}

pub fn asinh<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Asinh)
}

pub fn acos<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Acos)
}

pub fn acosh<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Acosh)
}

pub fn atan<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Atan)
}

pub fn atanh<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Atanh)
}

pub fn not<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::Not)
}

pub fn int<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::CastInt)
}

pub fn float<V>(x: V) -> Var
    where
        V: Variable,
{
    unary_map(x, UnaryMap::CastFloat)
}

// binary maps

pub fn add<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Add)
}

pub fn sub<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Sub)
}

pub fn div<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Div)
}

pub fn mul<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Mul)
}

pub fn modular<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Mod)
}

pub fn pow<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Pow)
}

pub fn min<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Min)
}

pub fn max<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Max)
}

pub fn and<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::And)
}

pub fn or<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Or)
}

pub fn eq<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Eq)
}

pub fn ne<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Ne)
}

pub fn gt<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Gt)
}

pub fn ge<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Ge)
}

pub fn lt<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Lt)
}

pub fn le<V1, V2>(x1: V1, x2: V2) -> Var
    where
        V1: Variable,
        V2: Variable,
{
    binary_map(x1, x2, BinaryMap::Le)
}

pub fn cond<V1, V2, V3>(c: V1, x1: V2, x2: V3) -> Var
    where
        V1: Variable,
        V2: Variable,
        V3: Variable,
{
    let (c, x1, x2) = broadcast3(c, x1, x2);

    assert_eq!(x1.data_type(), x2.data_type());
    assert_eq!(c.data_type(), DataType::Uint);

    Var::from_ternary_op(
        TernaryMapOperator {
            input: [c.desc().clone(), x1.desc().clone(), x2.desc().clone()],
            output: x1.desc().pristine(),
            map: TernaryMap::Cond,
            cached: Default::default(),
        },
        c,
        x1,
        x2,
    )
}

impl NaryOperator<0> for NullaryMapOperator {
    fn input(&self) -> &[TensorDesc; 0] {
        &[]
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: [&Var; 0], _: &Var, _: &Var) -> [Option<Var>; 0] {
        []
    }

    fn compute(&self, _: [&Tensor; 0], ctx: &mut Context) -> Result<Tensor, Error> {
        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        self.cached
            .load_program(ctx, || {
                let exp = match &self.map {
                    NullaryMap::Literal(val) => val.to_string(),
                };
                let data_type = y.data_type();
                format!(
                    r#"
            __kernel void nullary_map(__global {data_type} *y) {{
                const uint gid = get_global_id(0);
                y[gid] = {exp};
            }} "#
                )
            })
            .kernel("nullary_map")
            .arg_tensor(&y)
            .global_work_size(y.size())
            .launch()
            .map_err(Error::Device)?;
        Ok(y)
    }

    fn cat(&self) -> Category {
        Category::Map(MapOperator::Nullary(self.clone()))
    }
}

impl NaryOperator<1> for UnaryMapOperator {
    fn input(&self) -> &[TensorDesc; 1] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Var; 1], y: &Var, gy: &Var) -> [Option<Var>; 1] {
        let x = x[0];
        [match self.map {
            UnaryMap::Copy => Some(gy.clone()),
            UnaryMap::Abs => Some(sign(x) * gy),
            UnaryMap::Neg => Some(-gy.clone()),
            UnaryMap::Recip => Some(-gy / x.square()),
            UnaryMap::Log => Some(gy / x),
            UnaryMap::Exp => Some(gy * y),
            UnaryMap::Sqrt => Some(gy / (y * 2.0)),
            UnaryMap::Square => Some(gy * x * 2.0),
            UnaryMap::Erf => Some(gy * (2.0 / PI.sqrt()) / x.square().exp()),
            UnaryMap::Sign => Some(scalar(0.0, x.extents())),
            UnaryMap::Ceil => Some(scalar(0.0, x.extents())),
            UnaryMap::Floor => Some(scalar(0.0, x.extents())),
            UnaryMap::Round => Some(scalar(0.0, x.extents())),
            UnaryMap::Sin => Some(gy * x.cos()),
            UnaryMap::Sinh => Some(gy * x.cosh()),
            UnaryMap::Cos => Some(-gy * x.sin()),
            UnaryMap::Cosh => Some(gy * x.sinh()),
            UnaryMap::Tan => Some(gy / x.cos().square()),
            UnaryMap::Tanh => Some(gy / x.cosh().square()),
            UnaryMap::Asin => Some(gy / (-x.square() + 1.0).sqrt()),
            UnaryMap::Asinh => Some(gy / (x.square() + 1.0).sqrt()),
            UnaryMap::Acos => Some(-gy / (-x.square() + 1.0).sqrt()),
            UnaryMap::Acosh => Some(gy / (x.square() - 1.0).sqrt()),
            UnaryMap::Atan => Some(gy / (x.square() + 1.0)),
            UnaryMap::Atanh => Some(gy / (-x.square() + 1.0)),
            UnaryMap::Not => None,
            UnaryMap::CastInt | UnaryMap::CastFloat => Some(gy.clone()),
        }]
    }

    fn compute(&self, x: [&Tensor; 1], ctx: &mut Context) -> Result<Tensor, Error> {
        // let dtype = match self.op {
        //     UnaryMapOperator::Copy | UnaryMapOperator::Abs | UnaryMapOperator::Neg => x.data_type(),
        //     UnaryMapOperator::Sign | UnaryMapOperator::Not => DataType::Uint,
        //     UnaryMapOperator::CastInt => DataType::Int,
        //     UnaryMapOperator::CastFloat => DataType::Float,
        //     _ => DataType::Float,
        // };
        let x = x[0];
        let fn_name = "unary_map";
        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        self.cached
            .load_program(ctx, || {
                let s = compile_map(fn_name, &[x], self.map_fn("x0"), self.output());

                //println!("{:?}", &s);

                s
            })
            .kernel(fn_name)
            .arg_tensor(x)
            .arg_tensor(&y)
            .global_work_size(y.size())
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }

    fn cat(&self) -> Category {
        Category::Map(MapOperator::Unary(self.clone()))
    }
}

impl NaryOperator<2> for BinaryMapOperator {
    fn input(&self) -> &[TensorDesc; 2] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Var; 2], _: &Var, gy: &Var) -> [Option<Var>; 2] {
        match self.map {
            BinaryMap::Add => [Some(gy.clone()), Some(gy.clone())],
            BinaryMap::Sub => [Some(gy.clone()), Some(-gy.clone())],
            BinaryMap::Div => [Some(gy / x[1]), Some(gy * (-x[0] / x[1].square()))],
            BinaryMap::Mul => [Some(gy * x[1]), Some(gy * x[0])],
            BinaryMap::Mod => [Some(gy.clone()), Some(-gy * (x[0] / x[1]).floor())],
            BinaryMap::Pow => [
                Some(gy * x[1] * x[0].pow(x[1] - 1.0)),
                Some(gy * x[0].pow(x[1]) * x[0].log()),
            ],
            BinaryMap::Min => [
                Some(gy * lt(x[0], x[1]).float()),
                Some(gy * ge(x[0], x[1]).float()),
            ],
            BinaryMap::Max => [
                Some(gy * lt(x[1], x[0]).float()),
                Some(gy * ge(x[1], x[0]).float()),
            ],
            BinaryMap::And => [None, None],
            BinaryMap::Or => [None, None],
            BinaryMap::Eq => [None, None],
            BinaryMap::Ne => [None, None],
            BinaryMap::Gt => [None, None],
            BinaryMap::Ge => [None, None],
            BinaryMap::Lt => [None, None],
            BinaryMap::Le => [None, None],
        }
    }

    fn compute(&self, x: [&Tensor; 2], ctx: &mut Context) -> Result<Tensor, Error> {
        let fn_name = "binary_map";

        // (96, 128) (128, 1)
        // (256, 128) (0, 1)

        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        self.cached
            .load_program(ctx, || {
                compile_map(fn_name, &x, self.map_fn("x0", "x1"), self.output())
            })
            .kernel(fn_name)
            .arg_tensor(x[0])
            .arg_tensor(x[1])
            .arg_tensor(&y)
            .global_work_size(y.size())
            .launch()
            .map_err(Error::Device)?;

        // let p = if let Some(p) = ctx.cached_program("1") {
        //     p
        // } else {
        //     let src = compile_map(fn_name, &x, ctx, self.map_fn("x0", "x1"), self.output());
        //     ctx.get_program("1", src)
        // }
        //     .kernel(fn_name)
        //     .arg_tensor(x[0])
        //     .arg_tensor(x[1])
        //     .arg_tensor(&y)
        //     .global_work_size(y.size())
        //     .launch();

        Ok(y)
    }

    fn cat(&self) -> Category {
        Category::Map(MapOperator::Binary(self.clone()))
    }
}

impl NaryOperator<3> for TernaryMapOperator {
    fn input(&self) -> &[TensorDesc; 3] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, x: [&Var; 3], _: &Var, gy: &Var) -> [Option<Var>; 3] {
        match self.map {
            TernaryMap::Cond => [Some(cond(x[0], gy, 0.0)), Some(cond(x[0], 0.0, gy)), None],
            _ => [None, None, None],
        }
    }

    fn compute(&self, x: [&Tensor; 3], ctx: &mut Context) -> Result<Tensor, Error> {
        let fn_name = "ternary_map";

        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        self.cached
            .load_program(ctx, || {
                compile_map(fn_name, &x, self.map_fn("x0", "x1", "x2"), self.output())
            })
            .kernel(fn_name)
            .arg_tensor(x[0])
            .arg_tensor(x[1])
            .arg_tensor(x[2])
            .arg_tensor(&y)
            .global_work_size(y.size())
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }

    fn cat(&self) -> Category {
        Category::Map(MapOperator::Ternary(self.clone()))
    }
}

impl VariadicOperator for VariadicMapOperator {
    fn input(&self) -> &[TensorDesc] {
        &self.input
    }

    fn output(&self) -> &TensorDesc {
        &self.output
    }

    fn grad(&self, _: &[Var], _: &Var, _: &Var) -> Vec<Option<Var>> {
        // non-differentiable
        panic!("variadic map operators cannot be differentiated")
    }

    fn compute(&self, x: &[Tensor], ctx: &mut Context) -> Result<Tensor, Error> {
        let fn_name = "variadic";

        let y = Tensor::uninit2(self.output(), ctx)?; // all defaults to gpu tensors

        let p = self.cached.load_program(ctx, || {
            let exp = self.map_fn(&(0..x.len()).map(|i| format!("x{i}")).collect_vec());
            compile_map(fn_name, &x.iter().collect_vec(), exp, self.output())
        });

        //
        // let p = ctx.cached_program_or("1", || {
        //     let exp = self.map_fn(&(0..x.len()).map(|i| format!("x{i}")).collect_vec());
        //     compile_map(
        //         fn_name,
        //         &x.iter().collect_vec(),
        //         exp,
        //         self.output(),
        //     )
        // });

        //println!("{:?}", &src);
        let mut k = p.kernel(fn_name);

        for t in x {
            k = k.arg_tensor(t);
        }

        k.arg_tensor(&y)
            .global_work_size(y.size())
            .launch()
            .map_err(Error::Device)?;

        Ok(y)
    }

    fn cat(&self) -> Category {
        Category::Map(MapOperator::Variadic(self.clone()))
    }
}

///

impl<T> ops::Add<T> for Var
    where
        T: Variable,
{
    type Output = Var;
    fn add(self, x: T) -> Self::Output {
        add(self, x)
    }
}

impl<T> ops::Add<T> for &Var
    where
        T: Variable,
{
    type Output = Var;
    fn add(self, x: T) -> Self::Output {
        add(self, x)
    }
}

impl<T> ops::Add<T> for Tensor
    where
        T: Variable,
{
    type Output = Var;
    fn add(self, x: T) -> Self::Output {
        add(self, x)
    }
}

impl<T> ops::Add<T> for &Tensor
    where
        T: Variable,
{
    type Output = Var;
    fn add(self, x: T) -> Self::Output {
        add(self, x)
    }
}

/// SUB
///
///
impl<T> ops::Sub<T> for Var
    where
        T: Variable,
{
    type Output = Var;
    fn sub(self, x: T) -> Self::Output {
        sub(self, x)
    }
}

impl<T> ops::Sub<T> for &Var
    where
        T: Variable,
{
    type Output = Var;
    fn sub(self, x: T) -> Self::Output {
        sub(self, x)
    }
}

impl<T> ops::Sub<T> for Tensor
    where
        T: Variable,
{
    type Output = Var;
    fn sub(self, x: T) -> Self::Output {
        sub(self, x)
    }
}

impl<T> ops::Sub<T> for &Tensor
    where
        T: Variable,
{
    type Output = Var;
    fn sub(self, x: T) -> Self::Output {
        sub(self, x)
    }
}

/// MUL
///
impl<T> ops::Mul<T> for Var
    where
        T: Variable,
{
    type Output = Var;
    fn mul(self, x: T) -> Self::Output {
        mul(self, x)
    }
}

impl<T> ops::Mul<T> for &Var
    where
        T: Variable,
{
    type Output = Var;
    fn mul(self, x: T) -> Self::Output {
        mul(self, x)
    }
}

impl<T> ops::Mul<T> for &Tensor
    where
        T: Variable,
{
    type Output = Var;
    fn mul(self, x: T) -> Self::Output {
        mul(self, x)
    }
}

// Div

impl<T> ops::Div<T> for Var
    where
        T: Variable,
{
    type Output = Var;
    fn div(self, x: T) -> Self::Output {
        div(self, x)
    }
}

impl<T> ops::Div<T> for &Var
    where
        T: Variable,
{
    type Output = Var;
    fn div(self, x: T) -> Self::Output {
        div(self, x)
    }
}

impl<T> ops::Div<T> for Tensor
    where
        T: Variable,
{
    type Output = Var;
    fn div(self, x: T) -> Self::Output {
        div(self, x)
    }
}

impl<T> ops::Div<T> for &Tensor
    where
        T: Variable,
{
    type Output = Var;
    fn div(self, x: T) -> Self::Output {
        div(self, x)
    }
}

// Neg
impl ops::Neg for Var {
    type Output = Var;

    fn neg(self) -> Self::Output {
        neg(self)
    }
}

impl ops::Neg for &Var {
    type Output = Var;

    fn neg(self) -> Self::Output {
        neg(self)
    }
}

impl ops::Neg for Tensor {
    type Output = Var;

    fn neg(self) -> Self::Output {
        neg(self)
    }
}

impl ops::Neg for &Tensor {
    type Output = Var;

    fn neg(self) -> Self::Output {
        neg(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::core::{tr, transpose};
    use crate::ops::map::{add, cond, div, gt, max, min, modular, mul, neg, pow, scalar, sin, sub};
    use crate::session::context::Context;
    use crate::tensor::Tensor;
    use crate::var::{grad_check, Var};

    #[test]
    pub fn test_nullary_map() {
        let mut ctx = Context::new();

        let y_gt = Tensor::new([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]);

        let y = scalar(3.0, [3, 3]).eval(&mut ctx);

        assert!(Tensor::all_close(&y, &y_gt, 0.001));
    }

    #[test]
    pub fn test_unary_map() {
        let mut ctx = Context::new();
        let x = Tensor::new([
            [
                [0.5173, -0.9896, -0.7773],
                [2.1546, -0.7499, 0.2420],
                [-1.6632, 1.0712, -0.2654],
            ],
            [
                [-0.0449, -1.7201, 0.0733],
                [1.1641, -0.2699, 0.5033],
                [0.2659, 0.0322, 0.2114],
            ],
            [
                [0.2704, 1.0973, 0.3342],
                [-0.1980, -0.0689, 0.7259],
                [0.2048, -0.8022, 0.4763],
            ],
        ])
            .to_device(&mut ctx);

        let y_gt = Tensor::new([
            [
                [0.4945, -0.8358, -0.7013],
                [0.8344, -0.6816, 0.2396],
                [-0.9957, 0.8778, -0.2623],
            ],
            [
                [-0.0449, -0.9889, 0.0732],
                [0.9184, -0.2666, 0.4824],
                [0.2628, 0.0322, 0.2099],
            ],
            [
                [0.2671, 0.8900, 0.3280],
                [-0.1967, -0.0689, 0.6638],
                [0.2034, -0.7189, 0.4585],
            ],
        ]);

        let y = sin(&x).eval(&mut ctx);
        assert!(Tensor::all_close(&y, &y_gt, 0.001));

        let y = sin(&x.transpose(0, 1)).eval(&mut ctx);
        assert!(Tensor::all_close(&y, &y_gt.transpose(0, 1), 0.001));

        let y = sin(transpose(&x, 0, 1)).eval(&mut ctx);
        assert!(Tensor::all_close(&y, &y_gt.transpose(0, 1), 0.001));

        let x = Var::new(x);

        let y = x.abs();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = neg(&x);
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.recip();
        assert!(grad_check(&y, &x, 0.1, &mut ctx));

        let y = x.log();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.exp();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.sqrt();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.square();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.erf();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.sign();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.ceil();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.floor();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.round();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.sin();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.sinh();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.cos();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.cosh();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.tan();
        assert!(grad_check(&y, &x, 0.1, &mut ctx));

        let y = x.tanh();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.asin();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.asinh();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.acos();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.acosh();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.atan();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));

        let y = x.atanh();
        assert!(grad_check(&y, &x, 0.01, &mut ctx));
    }

    #[test]
    pub fn test_binary_map() {
        let mut ctx = Context::new();

        let x1 = Tensor::new([
            [
                [0.5999, -1.1316, -0.4251],
                [1.8613, 0.1228, 1.5092],
                [0.1597, 1.6246, -0.9158],
            ],
            [
                [-0.4136, -1.2864, 1.0150],
                [-0.1706, -0.1804, 0.3857],
                [0.3895, 0.5683, -0.7194],
            ],
            [
                [1.8544, -1.0580, -1.6796],
                [-1.1325, 0.5554, -0.7000],
                [1.3456, -0.0536, -1.2374],
            ],
        ])
            .to_device(&mut ctx);

        let x2 = Tensor::new([
            [
                [-0.4294, -0.9627, 0.9252],
                [-0.4139, 1.7050, 0.0189],
                [-1.1938, -0.5422, 0.7203],
            ],
            [
                [-0.1963, 0.2938, 0.5048],
                [-1.0904, 0.8356, 0.2139],
                [-1.3967, 1.0843, 0.5929],
            ],
            [
                [-0.4766, -0.2366, 1.6929],
                [1.3614, 1.4963, 1.0086],
                [0.8273, -1.7667, 0.7262],
            ],
        ])
            .to_device(&mut ctx);

        let y_gt = Tensor::new([
            [
                [1.0293, -0.1689, -1.3503],
                [2.2752, -1.5822, 1.4903],
                [1.3535, 2.1668, -1.6361],
            ],
            [
                [-0.2173, -1.5802, 0.5102],
                [0.9198, -1.0160, 0.1718],
                [1.7862, -0.5160, -1.3123],
            ],
            [
                [2.3310, -0.8214, -3.3725],
                [-2.4939, -0.9409, -1.7086],
                [0.5183, 1.7131, -1.9636],
            ],
        ]);

        let y = sub(&x1, &x2).eval(&mut ctx);
        assert!(Tensor::all_close(&y, &y_gt, 0.001));

        let y = sub(tr(&x1), tr(&x2)).eval(&mut ctx);
        assert!(Tensor::all_close(&y, &y_gt.transpose(0, 1), 0.001));

        // check gradients
        let x1 = Var::new(x1);
        let x2 = Var::new(x2);

        let y = add(&x1, &x2);
        assert!(grad_check(&y, &x1, 0.01, &mut ctx));
        assert!(grad_check(&y, &x2, 0.01, &mut ctx));

        let y = sub(&x1, &x2);
        assert!(grad_check(&y, &x1, 0.01, &mut ctx));
        assert!(grad_check(&y, &x2, 0.01, &mut ctx));

        // let y = div(&x1, &x2);
        // assert!(grad_check(&y, &x1, 0.01, &mut ctx));
        // assert!(grad_check(&y, &x2, 0.01, &mut ctx));

        let y = mul(&x1, &x2);
        assert!(grad_check(&y, &x1, 0.01, &mut ctx));
        assert!(grad_check(&y, &x2, 0.01, &mut ctx));

        // let y = modular(&x1, &x2);
        // assert!(grad_check(&y, &x1, 0.01, &mut ctx));
        // assert!(grad_check(&y, &x2, 0.01, &mut ctx));

        // let y = pow(&x1, &x2);
        // assert!(grad_check(&y, &x1, 0.01, &mut ctx));
        // assert!(grad_check(&y, &x2, 0.01, &mut ctx));

        let y = min(&x1, &x2);
        assert!(grad_check(&y, &x1, 0.01, &mut ctx));
        assert!(grad_check(&y, &x2, 0.01, &mut ctx));

        let y = max(&x1, &x2);
        assert!(grad_check(&y, &x1, 0.01, &mut ctx));
        assert!(grad_check(&y, &x2, 0.01, &mut ctx));
    }

    #[test]
    fn test_ternary_map() {
        let mut ctx = Context::new();

        let x1 = Tensor::new([
            [0.8975, -1.3578, 0.8378],
            [-2.2269, 0.1640, 0.4788],
            [0.1963, 1.7519, 1.1383],
        ])
            .to_device(&mut ctx);

        let x2 = Tensor::new([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]).to_device(&mut ctx);

        let y_gt = Tensor::new([
            [0.8975, 1.0000, 0.8378],
            [1.0000, 0.1640, 0.4788],
            [0.1963, 1.7519, 1.1383],
        ]);

        let y = cond(gt(&x1, 0.0), &x1, &x2).eval(&mut ctx);
        assert!(Tensor::all_close(&y, &y_gt, 0.001));
    }

    #[test]
    fn test_variadic_map() {
        let mut ctx = Context::new();

        let x1 = Tensor::new([
            [1.0701, 0.7608, 1.4853],
            [0.0172, 2.5665, 1.0956],
            [-2.5104, -1.6239, 0.6691],
        ])
            .to_device(&mut ctx);

        let x2 = Tensor::new([
            [-1.2409, 0.9614, -2.1566],
            [-0.8321, 1.9756, -0.1262],
            [0.8036, -1.7905, -0.1739],
        ])
            .to_device(&mut ctx);

        let x3 = Tensor::new([
            [0.3088, -0.1981, -0.0361],
            [-0.5297, 0.2176, -1.2164],
            [-0.1956, -1.0025, -1.0303],
        ])
            .to_device(&mut ctx);

        let y_gt = Tensor::new([
            [2.2621e+00, 3.8944e+00, 2.3029e+00],
            [2.6610e+00, 4.2441e+00, -1.0503e+01],
            [-2.2813e-03, 3.6270e+00, -3.8094e+00],
        ]);

        let y = (((&x1 + &x2 * 3.0) - &x3 / 2.0) / &x2).eval(&mut ctx);
        assert!(Tensor::all_close(&y, &y_gt, 0.01));
    }
}
