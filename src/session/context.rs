use crate::error::Error;
use crate::ops::Operator;
use crate::session::device::{Buffer, Device, KernelProgram};
use crate::session::memory;
use crate::session::memory::{Allocator, MemoryError};
use crate::shape::SizedExtent;
use crate::tensor::data::{DataType, OpenClData};
use crate::tensor::{Data, Tensor};
use crate::var::Var;
use crate::{session};
use itertools::Itertools;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use crate::session::reactor::Reactor;

extern crate ocl;

pub struct Context {
    device: Device,
    pub data: HashMap<Var, Tensor>,

    // compiled kernels
    kernels: Vec<KernelProgram>,
    kernel_idx_by_src: HashMap<String, usize>,
    // memory cap
    pub memory: Allocator,

    pub reactor: Arc<Mutex<Box<Reactor>>>,
}

impl Context {
    pub fn new() -> Self {
        Self::with_device(Device::get_first_gpu())
    }

    pub fn with_device(idx: usize) -> Self {
        Context {
            device: Device::new(idx),
            data: HashMap::new(),
            kernels: Vec::new(),
            kernel_idx_by_src: HashMap::new(),
            memory: Allocator::new(1000 * 1024 * 1024 / 4, false, true),
            reactor: Reactor::new(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn data(&self) -> &HashMap<Var, Tensor> {
        &self.data
    }

    pub fn get_program_with_id<S>(&mut self, src: S) -> (usize, &KernelProgram)
        where
            S: AsRef<str>,
    {
        if !self.kernel_idx_by_src.contains_key(src.as_ref()) {
            //println!("{}", src.as_ref());
            //            let now = Instant::now();

            let ker_prog = KernelProgram::new(src.as_ref().to_string(), self).unwrap();
            self.kernels.push(ker_prog);
            // let e2 = now.elapsed();
            // if e2.as_millis() > 30 {
            //     println!("{}", src.as_ref());
            //     println!(" elapsed: {:.2?}", e2);
            // }
            self.kernel_idx_by_src
                .insert(src.as_ref().to_string(), self.kernels.len() - 1);
        }

        let idx = *self.kernel_idx_by_src.get(src.as_ref()).unwrap();
        (idx, &self.kernels[idx])
    }

    pub fn get_program<S>(&mut self, src: S) -> &KernelProgram
        where
            S: AsRef<str>,
    {
        self.get_program_with_id(src).1
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn eval<'a, I>(&mut self, targets: I)
        where
            I: IntoIterator<Item=&'a Var>,
    {
        // get (already evaluated) inputs
        let y: Vec<&Var> = targets.into_iter().collect_vec();

        let mut stack = y.clone();
        let mut x = Vec::new();

        while !stack.is_empty() {
            let e = stack.pop().unwrap();
            if self.data.contains_key(e) {
                x.push(e);
            } else if e.is_op() {
                stack.extend(e.op().unwrap().input());
            }
        }

        let p = session::Program::compile(x, y);
        p.exec(self);
    }

    pub fn alloc(&mut self, data_type: DataType, size: usize) -> Result<OpenClData, Error> {
        // let v1 = self.m_t.values().map(|t| t.size()).sum::<usize>() * 4 / (1024 * 1024);
        //println!("alloc {} MB", self.memory.mem_used() * 4 / (1024 * 1024));

        //let before = self.memory.mem_used();
        let m = self
            .memory
            .alloc(size, &self.device)
            .map(|memory| OpenClData { memory, data_type });
        //let after = self.memory.mem_used();

        // println!(
        //     "!! mem_before: {} MB / mem_after (actual): {} MB",
        //     before * 4 / (1024 * 1024),
        //     after * 4 / (1024 * 1024)
        // );

        m
    }
}

// ctx
struct LastCall {
    ctx: *const Context,
    kernel_idx: usize,
}

#[derive(Default)]
pub struct CachedAccess {
    last_call: RefCell<Option<LastCall>>,
}

impl CachedAccess {
    pub fn new() -> Self {
        CachedAccess {
            last_call: RefCell::new(None),
        }
    }

    pub fn load_program<'a, F>(&self, ctx: &'a mut Context, gen_src: F) -> &'a KernelProgram
        where
            F: FnOnce() -> String,
    {
        let mut last_call = self.last_call.borrow_mut();
        let ctx_ptr = ctx as *const Context;

        if let Some(lc) = last_call.as_ref() {
            if lc.ctx == ctx_ptr {
                return &ctx.kernels[lc.kernel_idx];
            }
        }

        // create cache
        let (k_id, prog) = ctx.get_program_with_id(gen_src());

        *last_call = Some(LastCall {
            ctx: ctx_ptr,
            kernel_idx: k_id,
        });

        prog
    }

    pub fn is_empty(&self) -> bool {
        self.last_call.borrow().is_none()
    }
}

impl Clone for CachedAccess {
    fn clone(&self) -> Self {
        CachedAccess::new()
    }
}

impl Debug for CachedAccess {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "cached: {:?}", !self.is_empty())
    }
}
