use crate::error::Error;
use crate::session::context::Context;
use crate::session::device::Buffer;
use crate::session::memory;
use crate::session::memory::{OpenClMemory, MemoryError};
use std::cell::RefCell;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Mul;
use std::rc::Weak;

// OpenCL data types
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataType {
    // Integers
    Char,
    Short,
    Int,
    Long,

    // Unsigned Integers
    Uchar,
    Ushort,
    Uint,
    Ulong,

    // Floats
    Float,
    Half,
}

#[derive(Clone, Debug)]
pub enum HostData {
    // Integers
    Char(Vec<i8>),
    Short(Vec<i16>),
    Int(Vec<i32>),
    Long(Vec<i64>),

    // Unsigned Integers
    Uchar(Vec<u8>),
    Ushort(Vec<u16>),
    Uint(Vec<u32>),
    Ulong(Vec<u64>),

    // Floats
    Float(Vec<f32>),
    Half(Vec<f32>),
}

pub struct OpenClData {
    pub memory: OpenClMemory,
    pub data_type: DataType,
}

impl HostData {
    pub fn from_device(data: &OpenClData) -> Self {
        let buffer = data.buffer();

        match data.data_type {
            DataType::Char => HostData::Char(buffer.read::<i8>()),
            DataType::Short => HostData::Short(buffer.read::<i16>()),
            DataType::Int => HostData::Int(buffer.read::<i32>()),
            DataType::Long => HostData::Long(buffer.read::<i64>()),
            DataType::Uchar => HostData::Uchar(buffer.read::<u8>()),
            DataType::Ushort => HostData::Ushort(buffer.read::<u16>()),
            DataType::Uint => HostData::Uint(buffer.read::<u32>()),
            DataType::Ulong => HostData::Ulong(buffer.read::<u64>()),
            DataType::Float => HostData::Float(buffer.read::<f32>()),
            DataType::Half => unimplemented!(),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            HostData::Char(_) => DataType::Char,
            HostData::Short(_) => DataType::Short,
            HostData::Int(_) => DataType::Int,
            HostData::Long(_) => DataType::Long,
            HostData::Uchar(_) => DataType::Uchar,
            HostData::Ushort(_) => DataType::Ushort,
            HostData::Uint(_) => DataType::Uint,
            HostData::Ulong(_) => DataType::Ulong,
            HostData::Float(_) => DataType::Float,
            HostData::Half(_) => DataType::Half,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            HostData::Char(arr) => arr.len(),
            HostData::Short(arr) => arr.len(),
            HostData::Int(arr) => arr.len(),
            HostData::Long(arr) => arr.len(),
            HostData::Uchar(arr) => arr.len(),
            HostData::Ushort(arr) => arr.len(),
            HostData::Uint(arr) => arr.len(),
            HostData::Ulong(arr) => arr.len(),
            HostData::Float(arr) => arr.len(),
            HostData::Half(arr) => arr.len(),
        }
    }
}

impl OpenClData {
    pub fn new(size: usize, data_type: DataType, ctx: &mut Context) -> Result<Self, Error> {
        ctx.alloc(data_type, size)
    }

    pub fn from_host(data: &HostData, ctx: &mut Context) -> Self {
        let d = ctx.alloc(data.data_type(), data.len()).unwrap();
        let buffer = d.buffer();

        match data {
            HostData::Char(arr) => buffer.write(arr),
            HostData::Short(arr) => buffer.write(arr),
            HostData::Int(arr) => buffer.write(arr),
            HostData::Long(arr) => buffer.write(arr),
            HostData::Uchar(arr) => buffer.write(arr),
            HostData::Ushort(arr) => buffer.write(arr),
            HostData::Uint(arr) => buffer.write(arr),
            HostData::Ulong(arr) => buffer.write(arr),
            HostData::Float(arr) => buffer.write(arr),
            HostData::Half(arr) => {
                unimplemented!()
            }
        };

        d
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn len(&self) -> usize {
        self.buffer().size()
    }

    pub fn buffer(&self) -> &Buffer {
        self.memory.buffer()
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            DataType::Char => write!(f, "char"),
            DataType::Short => write!(f, "short"),
            DataType::Int => write!(f, "int"),
            DataType::Long => write!(f, "long"),
            DataType::Uchar => write!(f, "unsigned char"),
            DataType::Ushort => write!(f, "unsigned short"),
            DataType::Uint => write!(f, "unsigned int"),
            DataType::Ulong => write!(f, "unsigned long"),
            DataType::Float => write!(f, "float"),
            DataType::Half => write!(f, "half"),
        }
    }
}

impl DataType {
    pub fn bytes(&self) -> usize {
        match self {
            DataType::Char => 1,
            DataType::Short => 2,
            DataType::Int => 4,
            DataType::Long => 8,
            DataType::Uchar => 1,
            DataType::Ushort => 2,
            DataType::Uint => 4,
            DataType::Ulong => 8,
            DataType::Float => 4,
            DataType::Half => 2,
        }
    }

    pub fn opencl(&self) -> &'static str {
        match self {
            DataType::Char => "char",
            DataType::Short => "short",
            DataType::Int => "int",
            DataType::Long => "long",
            DataType::Uchar => "uchar",
            DataType::Ushort => "ushort",
            DataType::Uint => "uint",
            DataType::Ulong => "ulong",
            DataType::Float => "float",
            DataType::Half => "half",
        }
    }
}

pub trait Scalar: Sized + Copy + Send + Sync + Debug + Display + 'static {
    fn data_type() -> DataType;
    fn vec_to_data(v: Vec<Self>) -> HostData;
    fn data_to_vec(a: &HostData) -> &[Self];
}

impl<T> DataLiteral<T> for Vec<T>
where
    T: Scalar,
{
    fn flat_iter<'a>(&'a self) -> Box<dyn Iterator<Item = T> + 'a> {
        Box::new(self.iter().cloned())
    }

    fn extents(&self) -> Vec<usize> {
        vec![self.len()]
    }
}

pub trait DataLiteral<T>
where
    T: Scalar,
{
    fn flat_iter<'a>(&'a self) -> Box<dyn Iterator<Item = T> + 'a>;

    fn extents(&self) -> Vec<usize>;

    fn to_vec(&self) -> Vec<T> {
        self.flat_iter().collect()
    }

    fn to_buf(&self) -> HostData {
        T::vec_to_data(self.to_vec())
    }
}

impl<T, E, const C: usize> DataLiteral<T> for [E; C]
where
    E: DataLiteral<T>,
    T: Scalar,
{
    fn flat_iter<'a>(&'a self) -> Box<dyn Iterator<Item = T> + 'a> {
        Box::new(self.iter().flat_map(|a| a.flat_iter()))
    }

    fn extents(&self) -> Vec<usize> {
        let mut s = self[0].extents();
        s.insert(0, self.len());
        s
    }
}

impl<'a, E, T> DataLiteral<T> for &'a [E]
where
    E: DataLiteral<T>,
    T: Scalar,
{
    fn flat_iter<'b>(&'b self) -> Box<dyn Iterator<Item = T> + 'b> {
        Box::new(self.iter().flat_map(|a| a.flat_iter()))
    }

    fn extents(&self) -> Vec<usize> {
        let mut s = self[0].extents();
        s.insert(0, self.len());
        s
    }
}

impl<T> DataLiteral<T> for T
where
    T: Scalar,
{
    fn flat_iter(&self) -> Box<dyn Iterator<Item = T>> {
        Box::new(core::iter::once(*self))
    }

    fn extents(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl Scalar for f32 {
    fn data_type() -> DataType {
        DataType::Float
    }

    fn vec_to_data(v: Vec<f32>) -> HostData {
        HostData::Float(v)
    }

    fn data_to_vec(a: &HostData) -> &[Self] {
        if let HostData::Float(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}

impl Scalar for i64 {
    fn data_type() -> DataType {
        DataType::Long
    }

    fn vec_to_data(v: Vec<i64>) -> HostData {
        HostData::Long(v)
    }

    fn data_to_vec(a: &HostData) -> &[Self] {
        if let HostData::Long(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}

impl Scalar for i32 {
    fn data_type() -> DataType {
        DataType::Int
    }

    fn vec_to_data(v: Vec<i32>) -> HostData {
        HostData::Int(v)
    }

    fn data_to_vec(a: &HostData) -> &[Self] {
        if let HostData::Int(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}

impl Scalar for i16 {
    fn data_type() -> DataType {
        DataType::Short
    }

    fn vec_to_data(v: Vec<i16>) -> HostData {
        HostData::Short(v)
    }

    fn data_to_vec(a: &HostData) -> &[Self] {
        if let HostData::Short(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}

impl Scalar for i8 {
    fn data_type() -> DataType {
        DataType::Char
    }

    fn vec_to_data(v: Vec<i8>) -> HostData {
        HostData::Char(v)
    }

    fn data_to_vec(a: &HostData) -> &[Self] {
        if let HostData::Char(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}

impl Scalar for u64 {
    fn data_type() -> DataType {
        DataType::Ulong
    }

    fn vec_to_data(v: Vec<u64>) -> HostData {
        HostData::Ulong(v)
    }

    fn data_to_vec(a: &HostData) -> &[Self] {
        if let HostData::Ulong(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}

impl Scalar for u32 {
    fn data_type() -> DataType {
        DataType::Uint
    }

    fn vec_to_data(v: Vec<u32>) -> HostData {
        HostData::Uint(v)
    }

    fn data_to_vec(a: &HostData) -> &[Self] {
        if let HostData::Uint(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}

impl Scalar for u16 {
    fn data_type() -> DataType {
        DataType::Ushort
    }

    fn vec_to_data(v: Vec<u16>) -> HostData {
        HostData::Ushort(v)
    }

    fn data_to_vec(a: &HostData) -> &[Self] {
        if let HostData::Ushort(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}

impl Scalar for u8 {
    fn data_type() -> DataType {
        DataType::Uchar
    }

    fn vec_to_data(v: Vec<u8>) -> HostData {
        HostData::Uchar(v)
    }

    fn data_to_vec(a: &HostData) -> &[Self] {
        if let HostData::Uchar(v) = a {
            v
        } else {
            panic!("not convertable");
        }
    }
}
