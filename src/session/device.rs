// all opencl-related things go here

use crate::tensor::{Data, Tensor};
use itertools::{enumerate, Itertools};
use ocl::traits::WorkDims;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use crate::session::context::Context;

#[derive(Debug)]
pub struct DeviceError(ocl::Error);

impl DeviceError {
    pub fn from_core(e: ocl::core::Error) -> DeviceError {
        DeviceError(ocl::Error::from(e))
    }
}

pub struct Device {
    cl_context: ocl::Context,
    cl_device: ocl::Device,
    cl_queue: ocl::Queue,
}

impl Device {
    pub fn get_list() -> Vec<String> {
        let cl_platform = ocl::Platform::first().unwrap();
        let cl_devices = ocl::Device::list_all(cl_platform).unwrap();

        cl_devices
            .into_iter()
            .map(|d| d.name().unwrap())
            .collect_vec()
    }

    pub fn get_first_gpu() -> usize {
        for (i, d) in enumerate(Self::get_list()) {
            if d.to_lowercase().contains("nvidia") || d.to_lowercase().contains("radeon") {
                return i;
            }
        }
        0
    }

    pub fn new(idx: usize) -> Self {
        let cl_platform = ocl::Platform::first().unwrap();
        let cl_devices = ocl::Device::list_all(cl_platform).unwrap();

        let cl_context = ocl::Context::builder()
            .devices(ocl::Device::specifier().single(cl_devices[idx]))
            .build()
            .unwrap();

        let cl_device = cl_context.devices()[0];

        // for kk in cl_context.devices() {
        //     println!("{:?}", kk.name().unwrap());
        //     println!(
        //         "{:?}",
        //         ocl::core::get_device_info(&kk, DeviceInfo::MaxWorkItemSizes)
        //     );
        // }

        let cl_queue = ocl::Queue::new(&cl_context, cl_device, None).unwrap();

        Device {
            cl_context,
            cl_device,
            cl_queue,
        }
    }

    pub fn alloc_buffer(&self, size: usize) -> Result<Buffer, DeviceError> {
        let cl_mem = unsafe {
            ocl::core::create_buffer::<_, i32>(
                &self.cl_context,
                ocl::flags::MEM_READ_WRITE,
                size,
                None,
            )
                .map_err(|e| DeviceError::from_core(e))?
        };
        Ok(Buffer {
            cl_mem,
            queue: self.cl_queue.clone(),
            size,
            is_sub_buffer: false,
        })
    }
}

impl Debug for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.cl_device.name().unwrap())
    }
}

#[derive(Clone)]
pub struct Buffer {
    cl_mem: ocl::core::Mem,
    queue: ocl::Queue,
    size: usize,
    is_sub_buffer: bool,
}

impl Buffer {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn read<T>(&self) -> Vec<T>
        where
            T: ocl::OclPrm,
    {
        let mut vec = vec![T::default(); self.size]; // TODO size check 16bit vs 32bit
        unsafe {
            ocl::core::enqueue_read_buffer(
                &self.queue,
                &self.cl_mem,
                true,
                0,
                &mut vec,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
                .unwrap();
        }
        vec
    }

    pub fn write<T>(&self, data: &[T])
        where
            T: ocl::OclPrm,
    {
        unsafe {
            ocl::core::enqueue_write_buffer(
                &self.queue,
                &self.cl_mem,
                true,
                0,
                data,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
                .unwrap();
        }
    }

    pub fn fill<T>(&self, pattern: T)
        where
            T: ocl::OclPrm,
    {
        ocl::core::enqueue_fill_buffer(
            &self.queue,
            &self.cl_mem,
            pattern,
            0,
            self.size,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
            Some(&self.queue.device_version()),
        )
            .unwrap();
    }

    pub fn sub_buffer<T>(&self, size: usize, offset: usize) -> Buffer
        where
            T: ocl::OclPrm,
    {
        if self.is_sub_buffer() {
            panic!("cant create region of a region");
        }

        let cl_mem = ocl::core::create_sub_buffer::<T>(
            &self.cl_mem,
            ocl::flags::MemFlags::new(),
            &ocl::core::BufferRegion::new(offset, size),
        )
            .unwrap();
        Buffer {
            cl_mem,
            queue: self.queue.clone(),
            size,
            is_sub_buffer: true,
        }
    }

    pub fn is_sub_buffer(&self) -> bool {
        self.is_sub_buffer
    }
}

#[derive(Clone)]
pub struct KernelProgram {
    cl_program: ocl::Program,
    cl_queue: ocl::Queue,
    src: String,
}

impl KernelProgram {
    pub fn new(src: String, ctx: &Context) -> Result<Self, DeviceError> {
        let cl_program = ocl::Program::builder()
            .src(&src)
            .devices(&ctx.device().cl_device)
            .build(&ctx.device().cl_context)
            .map_err(|e| DeviceError(e))?;

        Ok(KernelProgram {
            cl_program,
            cl_queue: ctx.device().cl_queue.clone(),
            src,
        })
    }

    pub fn kernel(&self, name: &str) -> Kernel {
        let cl_kernel = ocl::core::create_kernel(&self.cl_program, name).unwrap();

        Kernel {
            cl_kernel,
            cl_queue: self.cl_queue.clone(),
            gws: None,
            lws: None,
            next_arg_idx: 0,
        }
    }

    pub fn src(&self) -> &str {
        self.src.as_str()
    }
}

pub struct Kernel {
    cl_kernel: ocl::core::Kernel,
    cl_queue: ocl::Queue,
    gws: Option<ocl::SpatialDims>,
    lws: Option<ocl::SpatialDims>,
    next_arg_idx: u32,
}

impl Kernel {
    pub fn arg_tensor<T>(mut self, val: T) -> Self
        where
            T: AsRef<Tensor>,
    {
        let data = val.as_ref().data();

        match data.deref() {
            Data::OpenCl(data) => {
                let buffer = data.buffer();

                let mem = &buffer.cl_mem;
                ocl::core::set_kernel_arg(
                    &self.cl_kernel,
                    self.next_arg_idx,
                    ocl::enums::ArgVal::mem(mem),
                )
                    .unwrap();
            }
            Data::Host(_) => panic!("tensor not on the backend!"),
            #[cfg(feature = "cuda")]
            Data::Cuda(_) => unimplemented!(),
        }
        self.next_arg_idx += 1;
        self
    }

    pub fn arg<T>(mut self, val: T) -> Self
        where
            T: ocl::OclPrm,
    {
        ocl::core::set_kernel_arg(
            &self.cl_kernel,
            self.next_arg_idx,
            ocl::enums::ArgVal::scalar(&val),
        )
            .unwrap();
        self.next_arg_idx += 1;
        self
    }

    pub fn global_work_size<E>(mut self, dims: E) -> Self
        where
            E: Into<ocl::SpatialDims>,
    {
        self.gws = Some(dims.into());
        self
    }

    pub fn local_work_size<E>(mut self, dims: E) -> Self
        where
            E: Into<ocl::SpatialDims>,
    {
        self.lws = Some(dims.into());
        self
    }

    pub fn launch(self) -> Result<(), DeviceError> {
        // core::enqueue_kernel(queue, &self.kernel, dim_count, self.gwo.to_work_offset(),
        //                      &gws, self.lws.to_work_size(), self.wait_events, self.new_event)
        //     .map_err(OclError::from)

        let gws = self.gws.unwrap();
        let lws = self.lws.map(|v| v.to_work_size().unwrap());

        unsafe {
            ocl::core::enqueue_kernel(
                &self.cl_queue,
                &self.cl_kernel,
                gws.dim_count(),
                None,
                &gws.to_work_size().unwrap(),
                lws,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
                .map_err(|e| DeviceError::from_core(e))
        }
    }
}

//
// pub struct KernelBuilder<'a> {
//     cl_builder: ocl::builders::KernelBuilder<'a>,
//     cl_queue: ocl::Queue,
// }
//
// impl<'a> KernelBuilder<'a> {
//     pub fn arg<A>(mut self, t: A) -> Self
//     where
//         A: ocl::OclPrm + 'a,
//     {
//         self.cl_builder.arg(t);
//         //t.arg(&mut self.cl_builder);
//         self
//     }
//
//     pub fn arg_tensor(mut self, t: &'a Tensor) -> Self {
//         let datac = t.data();
//         if let Data::Device(data) = datac.deref() {
//             self.cl_builder.arg()
//             // match data.buffer() {
//             //     DeviceBuffer::Char(buf) => self.cl_builder.arg(buf.clone()),
//             //     DeviceBuffer::Short(buf) => self.cl_builder.arg(buf.clone()),
//             //     DeviceBuffer::Int(buf) => self.cl_builder.arg(buf.clone()),
//             //     DeviceBuffer::Long(buf) => self.cl_builder.arg(buf.clone()),
//             //     DeviceBuffer::Uchar(buf) => self.cl_builder.arg(buf.clone()),
//             //     DeviceBuffer::Ushort(buf) => self.cl_builder.arg(buf.clone()),
//             //     DeviceBuffer::Uint(buf) => self.cl_builder.arg(buf.clone()),
//             //     DeviceBuffer::Ulong(buf) => self.cl_builder.arg(buf.clone()),
//             //     DeviceBuffer::Float(buf) => self.cl_builder.arg(buf.clone()),
//             //     DeviceBuffer::Half(buf) => unimplemented!(),
//             // };
//         } else {
//             panic!("tensor not on the backend!!!")
//         }
//         self
//     }
//
//     pub fn global_work_size<E>(mut self, dims: E) -> Self
//     where
//         E: Into<ocl::SpatialDims>,
//     {
//         self.cl_builder.global_work_size(dims);
//         self
//     }
//
//     pub fn local_work_size<E>(mut self, dims: E) -> Self
//     where
//         E: Into<ocl::SpatialDims>,
//     {
//         self.cl_builder.local_work_size(dims);
//         self
//     }
//
//     pub fn launch(mut self) {
//         let kernel = self
//             .cl_builder
//             .queue(self.cl_queue.clone())
//             .build()
//             .unwrap();
//
//         unsafe {
//             let res = kernel.enq();
//
//             res.unwrap();
//         }
//     }
// }
