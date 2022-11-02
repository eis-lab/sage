use itertools::Itertools;
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::Normal;
use rayon::prelude::*;
use std::borrow::Borrow;
use std::cell::{Ref, RefCell, RefMut};
use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use crate::tensor::data::{DataLiteral, DataType, OpenClData, HostData, Scalar};
use crate::tensor::iter::{AlongAxisIter, Iter};
use crate::backend;

pub mod data;
pub mod format;
pub mod init;
pub mod iter;

use crate::error::Error;
use crate::session::context::Context;
use crate::shape::{Axes, Axis, Extent, Shape};
use thiserror::Error;

#[derive(Error, Debug, Eq, PartialEq)]
pub enum BufferError {
    #[error("move tensor into host memory first")]
    HostMemory,

    #[error("Data type {} and {} are not compatible", .0, .1)]
    IncompatibleType(DataType, DataType),
}

#[derive(Clone)]
pub struct TensorDesc {
    pub shape: Shape,
    pub data_type: DataType,
}

impl TensorDesc {
    pub fn new<E>(extents: E, data_type: DataType) -> Self
        where
            E: Extent,
    {
        TensorDesc {
            shape: Shape::new(extents),
            data_type,
        }
    }

    pub fn pristine(&self) -> Self {
        TensorDesc::new(self.shape.extents(), self.data_type)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn extents(&self) -> &[usize] {
        self.shape.extents()
    }

    pub fn extent<A>(&self, axis: A) -> usize
        where
            A: Axis,
    {
        let axis = axis.to_usize(self.rank()).unwrap();
        self.shape.extents[axis]
    }

    pub fn rank(&self) -> usize {
        self.shape.num_axes()
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }
}

impl PartialEq<Self> for TensorDesc {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.data_type == other.data_type
    }
}

impl Eq for TensorDesc {}

impl Debug for TensorDesc {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({})",
            self.data_type.opencl(),
            self.extents().iter().map(|v| v.to_string()).join(", ")
        )
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub desc: TensorDesc,
    pub data: Arc<RefCell<Data>>,
}

impl Tensor {
    // ******************************** Constructors ******************************** //

    pub fn new<D, T>(data: D) -> Self
        where
            D: DataLiteral<T>,
            T: Scalar,
    {
        Tensor {
            desc: TensorDesc::new(data.extents(), T::data_type()),
            data: Arc::new(RefCell::new(Data::Host(data.to_buf()))),
        }
    }

    pub fn from_iter<S, I, T>(extents: S, iter: I) -> Self
        where
            S: Extent,
            I: Iterator<Item=T>,
            T: Scalar,
    {
        //let shape = shape.to_arr(0).unwrap();
        let data = T::vec_to_data(iter.collect());

        Tensor {
            desc: TensorDesc::new(extents, T::data_type()),
            data: Arc::new(RefCell::new(Data::Host(data))),
        }
    }

    pub fn from_vec<S, T>(extents: S, vec: Vec<T>) -> Self
        where
            S: Extent,
            T: Scalar,
    {
        //let shape = shape.to_arr(0).unwrap();
        let data = T::vec_to_data(vec);

        Tensor {
            desc: TensorDesc::new(extents, T::data_type()),
            data: Arc::new(RefCell::new(Data::Host(data))),
        }
    }

    pub fn from_scalar<E, T>(extents: E, val: T) -> Self
        where
            E: Extent,
            T: Scalar,
    {
        let shape = Shape::new(extents);
        let data = T::vec_to_data(vec![val; shape.size()]);

        Tensor {
            desc: TensorDesc {
                shape,
                data_type: T::data_type(),
            },
            data: Arc::new(RefCell::new(Data::Host(data))),
        }
    }

    pub fn from_dist<E, D, T>(extents: E, dist: D) -> Self
        where
            E: Extent,
            D: Distribution<T> + Sync,
            T: Scalar,
    {
        let extents = extents.to_arr(0).unwrap();

        let mut rng = thread_rng();
        let iter = (0..extents.iter().product())
            .into_par_iter()
            .map(|_| dist.sample(&mut thread_rng()));

        let data = T::vec_to_data(iter.collect());

        Tensor {
            desc: TensorDesc::new(extents, T::data_type()),
            data: Arc::new(RefCell::new(Data::Host(data))),
        }
    }

    // ******************************** Constructor Helpers ******************************** //

    pub fn uninit<S>(shape: S, data_type: DataType, ctx: &mut Context) -> Result<Self, Error>
        where
            S: Extent,
    {
        let shape = Shape::new(shape);
        let data = OpenClData::new(shape.size(), data_type, ctx)?;

        Ok(Tensor {
            desc: TensorDesc { shape, data_type },
            data: Arc::new(RefCell::new(Data::OpenCl(data))),
        })
    }

    pub fn uninit2(desc: &TensorDesc, ctx: &mut Context) -> Result<Self, Error> {
        Tensor::uninit(desc.extents(), desc.data_type(), ctx)
    }

    pub fn fill_zero(&self) {
        match self.data().deref() {
            Data::Host(data) => {
                panic!("only fill on backend tensors")
            }
            Data::OpenCl(data) => {
                data.buffer().fill(0);
            }
            #[cfg(feature = "cuda")]
            Data::Cuda(_) => unimplemented!(),
        };
    }

    pub fn zeros<S>(shape: S) -> Self
        where
            S: Extent,
    {
        Tensor::from_scalar(shape, 0.0)
    }

    pub fn ones<S>(shape: S) -> Self
        where
            S: Extent,
    {
        Tensor::from_scalar(shape, 1.0)
    }

    // sample from standard normal dist
    pub fn randn<S>(shape: S) -> Self
        where
            S: Extent,
    {
        Tensor::from_dist(shape, Normal::new(0.0, 1.0).unwrap())
    }

    // ******************************** Upload/Download ******************************** //

    pub fn is_host(&self) -> bool {
        let data = self.data();
        matches!(data.deref(), Data::Host(_))
    }

    pub fn is_device(&self) -> bool {
        !self.is_host()
    }

    pub fn device(&self, ctx: &mut Context) {
        let mut data = self.data_mut();
        if let Data::Host(d) = data.deref_mut() {
            *data = Data::OpenCl(OpenClData::from_host(d, ctx));
        }
    }

    pub fn host(&self) {
        let mut data = self.data_mut();
        if let Data::OpenCl(d) = data.deref_mut() {
            *data = Data::Host(HostData::from_device(d));
        }
    }

    pub fn set_data<D, T>(&self, data: D)
        where
            D: DataLiteral<T>,
            T: Scalar,
    {
        *self.data_mut().deref_mut() = Data::Host(data.to_buf());
    }

    pub fn to_device(&self, ctx: &mut Context) -> Tensor {
        match self.data().deref() {
            Data::Host(data) => {
                let data = OpenClData::from_host(data, ctx);
                Tensor {
                    desc: self.desc.clone(),
                    data: Arc::new(RefCell::new(Data::OpenCl(data))),
                }
            }
            Data::OpenCl(_) => self.clone(),
            #[cfg(feature = "cuda")]
            Data::Cuda(_) => unimplemented!(),
        }
    }

    pub fn to_host(&self) -> Tensor {
        match self.data().deref() {
            Data::Host(_) => self.clone(),
            Data::OpenCl(data) => {
                let data = HostData::from_device(data);

                Tensor {
                    desc: self.desc.clone(),
                    data: Arc::new(RefCell::new(Data::Host(data))),
                }
            }
            #[cfg(feature = "cuda")]
            Data::Cuda(_) => unimplemented!(),
        }
    }

    pub fn scalar<T>(&self) -> T
        where
            T: Scalar,
    {
        let buffer = self.buffer::<T>().unwrap();
        buffer.borrow()[0]
    }

    // ******************************** Tests ******************************** //
    pub fn all_close(t1: &Tensor, t2: &Tensor, eps: f32) -> bool {
        let (t1, t2) = (t1.to_host(), t2.to_host());

        (t1.size() == t2.size())
            && t1
            .iter::<f32>()
            .zip(t2.iter::<f32>())
            .all(|((_, v1), (_, v2))| (v1 - v2).abs() < eps)
    }

    pub fn all_equal<S>(t1: &Tensor, t2: &Tensor) -> bool
        where
            S: Scalar + Eq,
    {
        let (t1, t2) = (t1.to_host(), t2.to_host());

        (t1.size() == t2.size())
            && t1
            .iter::<S>()
            .zip(t2.iter::<S>())
            .all(|((_, v1), (_, v2))| v1 == v2)
    }

    // ******************************** Getters ******************************** //

    pub fn data(&self) -> Ref<Data> {
        RefCell::borrow(&self.data)
    }

    pub fn data_mut(&self) -> RefMut<Data> {
        RefCell::borrow_mut(&self.data)
    }

    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    pub fn buffer<T>(&self) -> Option<Ref<[T]>>
        where
            T: Scalar,
    {
        let data = self.data();

        if self.is_host() {
            Some(Ref::map(data, |d| {
                if let Data::Host(v) = d {
                    T::data_to_vec(v)
                } else {
                    panic!("ss")
                }
            }))
        } else {
            None
        }
    }

    pub fn data_type(&self) -> DataType {
        self.desc.data_type()
    }

    pub fn extents(&self) -> &[usize] {
        self.desc.extents()
    }

    pub fn extent<A>(&self, axis: A) -> usize
        where
            A: Axis,
    {
        self.desc.extent(axis)
    }

    pub fn strides(&self) -> &[usize] {
        self.desc.shape.strides()
    }

    pub fn offset(&self) -> usize {
        self.desc.shape.offset()
    }

    // alias of self.extents()
    pub fn shape(&self) -> &Shape {
        self.desc.shape()
    }

    pub fn rank(&self) -> usize {
        self.desc.rank()
    }

    pub fn size(&self) -> usize {
        self.desc.size()
    }

    // ******************************** Converters ******************************** //

    pub fn to_vec<T>(&self) -> Vec<T>
        where
            T: Scalar,
    {
        self.iter::<T>().map(|(_, v)| v).collect()
    }

    // ******************************** Iterators ******************************** //

    pub fn iter<T>(&self) -> Iter<T>
        where
            T: Scalar,
    {
        Iter::new(self)
    }

    pub fn along_axis<A, T>(&self, axis: A) -> AlongAxisIter
        where
            A: Axis,
            T: Scalar,
    {
        let axis = axis.to_usize(self.rank()).unwrap();
        AlongAxisIter::new(self, axis)
    }

    // ******************************** Memory layout ******************************** //

    fn with_layout(&self, layout: Shape) -> Tensor {
        Tensor {
            desc: TensorDesc {
                shape: layout,
                data_type: self.data_type(),
            },
            data: self.data.clone(),
        }
    }

    pub fn squeeze_axis<A>(&self, axis: A) -> Tensor
        where
            A: Axis,
    {
        let axis = axis.to_usize(self.rank()).unwrap();

        if self.extents()[axis] != 1 {
            panic!("only size=1 axes can be squeezed");
        }

        self.with_layout(self.shape().remove(axis))
    }

    pub fn expand_axis<A>(&self, axis: A) -> Tensor
        where
            A: Axis,
    {
        // allow non-existing index
        let axis = axis.to_usize(self.rank() + 1).unwrap();

        self.with_layout(self.shape().insert(axis))
    }

    // reshape (underlying data does not change)
    pub fn view<S>(&self, shape: S) -> Tensor
        where
            S: Extent,
    {
        let layout = Shape::new(shape);

        if self.shape().has_default_strides() {
            self.with_layout(layout)
        } else {
            panic!("Var::reorder() and try again");
            // create a new copy (with default memory layouts)
            //self.clone().with_layout(layout) // TODO: deep copy
        }
    }

    // swap last two dims of tensor
    pub fn transpose<A1, A2>(&self, axis1: A1, axis2: A2) -> Tensor
        where
            A1: Axis,
            A2: Axis,
    {
        let axis1 = axis1.to_usize(self.rank()).unwrap();
        let axis2 = axis2.to_usize(self.rank()).unwrap();

        if axis1 == axis2 {
            panic!("same axis");
        }

        self.with_layout(self.shape().swap(axis1, axis2))
    }

    pub fn permute<As>(&self, axes: As) -> Tensor
        where
            As: Axes,
    {
        let axes = axes.to_arr(self.rank()).unwrap();

        let mut use_counts = vec![0; self.rank()];

        axes.iter().for_each(|axis| {
            use_counts[*axis] += 1;
        });

        if use_counts.iter().any(|count| *count != 1) {
            panic!("some axes are not used, or used more than once");
        }

        self.with_layout(self.shape().permute(&axes))
    }

    pub fn expand<S>(&self, extent: S) -> Tensor
        where
            S: Extent,
    {
        let shape = extent.to_arr(0).unwrap();

        self.with_layout(self.shape().expand(&shape))
    }

    // ******************************** Indexing operations ******************************** //

    pub fn index<A>(&self, index: usize, axis: A) -> Tensor
        where
            A: Axis,
    {
        let axis = axis.to_usize(self.rank()).unwrap();
        let axis_size = self.extents()[axis];

        if axis_size <= index {
            panic!("index out of bounds");
        }

        self.with_layout(self.shape().select(index, axis))
    }

    pub fn slice<A>(&self, start: usize, end: usize, axis: A) -> Tensor
        where
            A: Axis,
    {
        let axis = axis.to_usize(self.rank()).unwrap();
        let axis_size = self.extents()[axis];

        if start >= end {
            panic!("start and end index are not in the order");
        }

        if axis_size < end {
            panic!("index out of bounds");
        }

        self.with_layout(self.shape().select_range(start, end - 1, axis))
    }
}

pub enum Data {
    Host(HostData),
    OpenCl(OpenClData),
    #[cfg(feature = "cuda")]
    Cuda(backend::cuda::Buffer),
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}
