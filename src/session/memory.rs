use crate::error::Error;
use crate::session::device::{Buffer, Device};
use crate::tensor::data::OpenClData;
use itertools::Itertools;
use ocl::enums::DeviceSpecifier::All;
use std::cell::{Cell, RefCell};
use std::cmp::{max, min, Ordering};
use std::collections::{BinaryHeap, LinkedList};
use std::fmt::{Debug, Formatter};
use std::ops::DerefMut;
use std::rc::{Rc, Weak};

/**

create mempool

request size

c.f., CuPy: https://github.com/cupy/cupy/blob/v10.4.0/cupy/cuda/memory.pyx
c.f., PyTorch https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDACachingAllocator.cpp
 **/

// all sizes are rounded to at least 512 bytes
static MIN_BLOCK_SIZE: usize = 512 / 4;

// largest "small" allocation is 1 MiB
static SMALL_SIZE: usize = 1048576 / 4;

// "small" allocations are packed in 2 MiB blocks
static SMALL_BUFFER: usize = 2097152 / 4;

// "large" allocations may be packed in 20 MiB blocks
static LARGE_BUFFER: usize = 20971520 / 4;

// allocations between 1 and 10 MiB may use kLargeBuffer
static MIN_LARGE_ALLOC: usize = 10485760 / 4;

// round up large allocations to 2 MiB
static ROUND_LARGE: usize = 2097152 / 4;

static SPLIT_LIM: usize = LARGE_BUFFER * 2; //200 * 1048576 / 32;

static GC_THRESHOLD: f32 = 0.8;

#[derive(thiserror::Error, Debug)]
pub enum MemoryError {
    #[error(
        "tried to allocate {mem_needed} bytes but only bytes are available out of {mem_total}"
    )]
    OutOfMemory { mem_needed: usize, mem_total: usize },

    #[error("no free block left in the pool")]
    NoFreeBlock,
}

struct Stat {
    cap: Option<usize>,
    total: usize,
    used: usize,
    peak: usize,
}

pub struct AllocatorOld {
    mem_cap: Option<usize>,
    stat: Rc<RefCell<Stat>>,
    small: Rc<RefCell<PoolOld>>,
    large: Rc<RefCell<PoolOld>>,
}
//
// impl AllocatorOld {
//     pub fn new(mem_cap: Option<usize>, use_cache: bool) -> Self {
//         let stat = Rc::new(RefCell::new(Stat {
//             cap: mem_cap,
//             total: 0,
//             used: 0,
//             peak: 0,
//         }));
//
//         AllocatorOld {
//             mem_cap,
//             stat,
//             small: Rc::new(RefCell::new(PoolOld::new(
//                 MIN_BLOCK_SIZE,
//                 SMALL_SIZE,
//                 use_cache,
//                 false,
//             ))),
//             large: Rc::new(RefCell::new(PoolOld::new(
//                 SMALL_SIZE,
//                 MIN_LARGE_ALLOC,
//                 use_cache,
//                 false,
//             ))),
//         }
//     }
//
//     pub fn alloc(&mut self, size: usize, backend: &Device) -> Result<Memory, Error> {
//         let pool = if size <= SMALL_SIZE {
//             &self.small
//         } else {
//             &self.large
//         };
//
//         let mem_curr = self.mem_used();
//         let block = { RefCell::borrow_mut(pool).deref_mut().alloc(size, backend)? };
//
//         if let Some(mem_cap) = self.mem_cap {
//             let mem_now = self.mem_used();
//             if mem_now > mem_cap {
//                 return Err(Error::Memory(MemoryError::OutOfMemory {
//                     mem_needed: mem_now - mem_curr,
//                     mem_total: mem_cap,
//                 }));
//             }
//         }
//
//         Ok(Memory {
//             pool: Rc::downgrade(pool),
//             block,
//             buffer: block.buffer(),
//         })
//     }
//
//     pub fn garbage_collect(&mut self, target_size: usize) -> usize {
//         let mut large = RefCell::borrow_mut(&self.large);
//         let mut small = RefCell::borrow_mut(&self.small);
//
//         let freed_size = large.deref_mut().garbage_collect(target_size);
//         let freed_size = small.deref_mut().garbage_collect(target_size - freed_size);
//
//         freed_size
//     }
//
//     pub fn mem_cap(&self) -> Option<usize> {
//         self.mem_cap
//     }
//
//     pub fn mem_used(&self) -> usize {
//         let large = RefCell::borrow(&self.large);
//         let small = RefCell::borrow(&self.small);
//
//         large.mem_used + small.mem_used
//     }
//
//     pub fn mem_peak(&self) -> usize {
//         let large = RefCell::borrow(&self.large);
//         let small = RefCell::borrow(&self.small);
//
//         large.mem_peak + small.mem_peak
//     }
// }
//
// impl Debug for AllocatorOld {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let mut large = RefCell::borrow(&self.large);
//         let mut small = RefCell::borrow(&self.small);
//
//         writeln!(
//             f,
//             "small pool: {}/{}",
//             self.small.borrow().mem_used,
//             self.small.borrow().mem_total
//         );
//         writeln!(
//             f,
//             "large pool: {}/{}",
//             self.large.borrow().mem_used,
//             self.large.borrow().mem_total
//         )
//     }
// }

pub struct Allocator {
    pool: Rc<RefCell<Pool>>,
}

impl Allocator {
    pub fn new(capacity: usize, use_cache: bool, use_sub_buffer: bool) -> Self {
        Allocator {
            pool: Rc::new(RefCell::new(Pool::new(capacity, use_cache, use_sub_buffer))),
        }
    }

    pub fn capacity(&self) -> usize {
        RefCell::borrow(&self.pool).capacity
    }

    pub fn mem_total(&self) -> usize {
        RefCell::borrow(&self.pool).mem_total
    }

    pub fn mem_peak(&self) -> usize {
        RefCell::borrow(&self.pool).mem_peak
    }

    pub fn mem_used(&self) -> usize {
        RefCell::borrow(&self.pool).mem_used
    }

    pub fn alloc(&self, size: usize, device: &Device) -> Result<OpenClMemory, Error> {
        let block = RefCell::borrow_mut(&self.pool).alloc(size, device)?;
        let buffer = block.buffer();
        Ok(OpenClMemory {
            pool: Rc::downgrade(&self.pool),
            block,
            buffer,
        })
    }
}

struct Pool {
    blocks_small: Vec<Block>,
    blocks_large: Vec<Block>,

    capacity: usize,

    mem_total: usize,
    mem_used: usize,
    mem_peak: usize,
    use_cache: bool,
    use_sub_buffer: bool,
}

impl Pool {
    fn new(capacity: usize, use_cache: bool, use_sub_buffer: bool) -> Self {
        Pool {
            blocks_small: Vec::new(),
            blocks_large: Vec::new(),
            capacity,
            mem_total: 0,
            mem_used: 0,
            mem_peak: 0,
            use_cache,
            use_sub_buffer,
        }
    }

    fn select_pool(&mut self, size: usize) -> &mut Vec<Block> {
        if size < SMALL_SIZE {
            &mut self.blocks_small
        } else {
            &mut self.blocks_large
        }
    }

    fn check_space(&self, size: usize) -> Result<(), Error> {
        if self.mem_total + size > self.capacity {
            Err(Error::Memory(MemoryError::OutOfMemory {
                mem_needed: (self.mem_total + size) - self.capacity,
                mem_total: self.mem_total,
            }))
        } else {
            Ok(())
        }
    }

    fn allocation_size(size: usize) -> usize {
        if size < SMALL_SIZE {
            SMALL_BUFFER
        } else if size < MIN_LARGE_ALLOC {
            LARGE_BUFFER
        } else {
            ROUND_LARGE * ((size + ROUND_LARGE - 1) / ROUND_LARGE)
        }
    }

    fn expand(&mut self, size: usize, device: &Device) -> Result<(), Error> {
        let alloc_size = Self::allocation_size(size);
        self.check_space(alloc_size)?;

        let pool = self.select_pool(size);

        let block = Block::new(device.alloc_buffer(alloc_size).map_err(Error::Device)?);
        pool.push(block);

        self.mem_total += alloc_size;
        if self.mem_total > self.mem_peak {
            self.mem_peak = self.mem_total;
        }
        Ok(())
    }

    fn get_free_block(&mut self, size: usize) -> Result<Block, MemoryError> {
        let use_sub_buffer = self.use_sub_buffer;
        let pool = self.select_pool(size);

        pool.iter().for_each(|b| b.age.up());

        let avail_size = pool
            .iter()
            .filter(|b| !b.is_split())
            .map(|b| b.size)
            .sum::<usize>();

        // println!(
        //     "--- total avail: {} MB, requested: {} MB, small: {} MB",
        //     avail_size * 4 / (1024 * 1024),
        //     size * 4 / (1024 * 1024),
        //     SMALL_SIZE * 4 / (1024 * 1024),
        // );

        let block_idx = pool
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                size <= b.size
                    && !(!use_sub_buffer
                        // Do not return an oversized block for a large request
                        || (size < SPLIT_LIM && SPLIT_LIM <= b.size)
                        // Allow oversized block size to be rounded up but within a limit
                        || (SPLIT_LIM <= size && size + LARGE_BUFFER <= b.size))
            })
            .min_by_key(|(_, b)| b.size)
            .map(|(i, _)| i)
            .ok_or(MemoryError::NoFreeBlock)?;

        let free_block = pool.remove(block_idx);
        free_block.age.reset();

        let remaining = free_block.size - size;
        let should_split = (size <= SMALL_SIZE && MIN_BLOCK_SIZE <= remaining)
            || (SMALL_SIZE < size && size < SPLIT_LIM && SMALL_SIZE < remaining);

        if should_split && use_sub_buffer {
            let (new, left) = free_block.split(size).unwrap();
            pool.push(left);

            self.mem_used += new.size;
            Ok(new)
        } else {
            self.mem_used += free_block.size;
            Ok(free_block)
        }
    }

    fn alloc(&mut self, size: usize, device: &Device) -> Result<Block, Error> {
        if self.use_cache {
            self.get_free_block(size).or_else(|_| {
                let freed_gc = self.garbage_collect();

                //println!("freed by gc: {} MB", freed_gc * 4 / (1024 * 1024));

                self.expand(size, device).or_else(|_| {
                    let freed_release = self.release_cached_blocks(size);

                    //println!("freed by release: {} MB", freed_release * 4 / (1024 * 1024));

                    self.expand(size, device)
                })?;

                self.get_free_block(size).map_err(Error::Memory)
            })
        } else {
            self.check_space(size)?;
            let block = Block::new(device.alloc_buffer(size).map_err(Error::Device)?);
            self.mem_used += block.size;
            self.mem_total += block.size;
            if self.mem_total > self.mem_peak {
                self.mem_peak = self.mem_total;
            }
            Ok(block)
        }
    }

    fn free(&mut self, block: &Block) {
        if self.use_cache {
            let use_sub_buffer = self.use_sub_buffer;
            let pool = self.select_pool(block.cap);

            if use_sub_buffer {
                let adjacent_blocks = pool
                    .drain_filter(|b| {
                        b.mem_eq(block)
                            && (b.offset + b.size == block.offset
                                || block.offset + block.size == b.offset)
                    })
                    .collect_vec();

                let (offset, size) = adjacent_blocks
                    .iter()
                    .fold((block.offset, block.size), |s, b| {
                        (min(s.0, b.offset), max(s.1, b.size))
                    });

                let mut new_block = block.clone();
                new_block.age.reset();
                new_block.offset = offset;
                new_block.size = size;

                // println!(
                //     "freed block is split? : {}, {}, {}, {}",
                //     offset,
                //     size,
                //     adjacent_blocks.len(),
                //     new_block.is_split()
                // );

                pool.push(new_block);
            } else {
                pool.push(block.clone());
            }

            self.mem_used -= block.size;
        } else {
            self.mem_used -= block.size;
            self.mem_total -= block.size
        }
    }

    fn garbage_collect(&mut self) -> usize {
        let gc_threshold_size = (self.capacity as f32 * GC_THRESHOLD) as usize;

        if self.mem_used < gc_threshold_size {
            return 0;
        }

        let target_size = self.capacity - gc_threshold_size;

        let mut total_age = 0;
        let mut freeable_blocks = 0;

        for block in self.blocks_large.iter() {
            if !block.is_split() {
                total_age += block.age.val();
                freeable_blocks += 1;
            }
        }

        if freeable_blocks == 0 {
            return 0;
        }

        let mut gc_reclaimed = 0;
        let mut block_freed = true;

        while gc_reclaimed < target_size && block_freed && freeable_blocks > 0 {
            let age_threshold = total_age / freeable_blocks;

            let gc_list = self
                .blocks_large
                .drain_filter(|b| !b.is_split() && b.age.val() >= age_threshold)
                .collect_vec();

            gc_reclaimed += gc_list.iter().map(|b| b.size).sum::<usize>();
            total_age -= gc_list.iter().map(|b| b.age.val()).sum::<usize>();

            block_freed = !gc_list.is_empty();
            freeable_blocks -= gc_list.len();
        }

        gc_reclaimed
    }

    fn release_cached_blocks(&mut self, size: usize) -> usize {
        let mut total_released = 0;

        while total_released < max(size, SPLIT_LIM) {
            if let Some(i) = self
                .blocks_large
                .iter()
                .enumerate()
                .filter(|(_, b)| !b.is_split())
                .max_by_key(|(_, b)| b.size)
                .map(|(i, _)| i)
            {
                let b = self.blocks_large.remove(i);
                total_released += b.size;
            } else {
                break;
            }
        }

        total_released
    }
}

struct PoolOld {
    blocks: LinkedList<Block>,
    min_size: usize,
    max_size: usize,

    mem_total: usize,
    mem_used: usize,
    mem_peak: usize,
    use_cache: bool,
    use_sub_buffer: bool,
}

impl PoolOld {
    fn new(min_size: usize, max_size: usize, use_cache: bool, use_sub_buffer: bool) -> Self {
        PoolOld {
            blocks: LinkedList::new(),
            min_size,
            max_size,
            mem_total: 0,
            mem_used: 0,
            mem_peak: 0,
            use_cache,
            use_sub_buffer,
        }
    }

    // fn alloc<F>(&mut self, size: usize, af: F) -> Option<Block>
    // where
    //     F: FnOnce(usize) -> Buffer,
    // {
    //     // if no free block?
    //
    //     // try merge ---> retry
    //     // still no?
    //     // alloc new buffer ---> retry
    //     // still no?
    //     // pool gc ---> retry
    //     // still no? --> OOM error
    //
    //     self.get_free_block(size).or_else(|| {
    //         let cap = max(self.max_size * 2, size);
    //
    //         let new_block = Block::new(af(cap));
    //         self.blocks.push_front(new_block);
    //
    //         self.get_free_block(size)
    //     })
    // }

    fn expand(&mut self, size: usize, device: &Device) -> Result<(), Error> {
        let cap = if self.use_cache && self.use_sub_buffer {
            max(self.max_size * 2, size)
        } else {
            size
        };

        let block = Block::new(device.alloc_buffer(cap).map_err(|e| Error::Device(e))?);
        self.blocks.push_front(block);

        self.mem_total += size;
        if self.mem_total > self.mem_peak {
            self.mem_peak = self.mem_total;
        }
        Ok(())
    }

    fn get_free_block(&mut self, size: usize) -> Result<Block, MemoryError> {
        // // find smallest block that fits.
        // let min_size = self
        //     .blocks
        //     .iter()
        //     .filter(|x| x.size >= size)
        //     .map(|x| x.size)
        //     .min()

        // get min size
        let mut min_size = None;
        for block in self.blocks.iter() {
            block.age.up();
            if block.size > size || min_size.is_none() {
                min_size = Some(block.size);
            }
        }

        let min_size = min_size.ok_or(MemoryError::NoFreeBlock)?;

        let mut cur = self.blocks.cursor_front_mut();

        while let Some(block) = cur.current() {
            if block.size == min_size {
                break;
            }
            cur.move_next();
        }
        let b = cur.remove_current().unwrap();

        if self.use_cache
            && self.use_sub_buffer
            && b.size >= size + self.min_size
            && size < SPLIT_LIM
        {
            let (new, left) = b.split(size).unwrap();
            self.mem_used += new.size;

            cur.insert_before(left);
            Ok(new)
        } else {
            self.mem_used += b.size;

            Ok(b)
        }
    }

    fn alloc(&mut self, size: usize, device: &Device) -> Result<Block, Error> {
        let size = max(size, self.min_size);

        if self.use_cache {
            self.get_free_block(size).or_else(|_| {
                self.expand(size, device)?;
                self.get_free_block(size).map_err(|e| Error::Memory(e))
            })
        } else {
            let block = Block::new(device.alloc_buffer(size).map_err(|e| Error::Device(e))?);
            self.mem_used += block.size;
            self.mem_total += block.size;
            if self.mem_total > self.mem_peak {
                self.mem_peak = self.mem_total;
            }
            Ok(block)
        }

        //
        // let free_block = self.get_free_block(size);
        //
        // if let Some(block) = free_block {
        //     Some(block)
        // } else {
        //     self.expand(size, backend);
        //     self.get_free_block(size)
        // }
    }

    fn free(&mut self, block: &Block) {
        if self.use_cache {
            let mut merged = false;
            let mut cur = self.blocks.cursor_front_mut();

            if self.use_sub_buffer {
                while let Some(cand) = cur.current() {
                    // look for adjacent blocks
                    if block.mem_eq(cand) {
                        if cand.offset + cand.size == block.offset {
                            cand.size += block.size;
                            merged = true;
                            break;
                        } else if block.offset + block.size == cand.offset {
                            cand.offset -= block.size;
                            merged = true;
                            break;
                        }
                    }
                    cur.move_next();
                }
            }

            if !merged {
                self.blocks.push_front(block.clone());
            }

            self.mem_used -= block.size;
        } else {
            self.mem_used -= block.size;
            self.mem_total -= block.size
        }
    }

    fn garbage_collect(&mut self, target_size: usize) -> usize {
        let mut cur = self.blocks.cursor_front_mut();
        let mut freed_size = 0;

        while freed_size < target_size {
            if let Some(block) = cur.current() {
                if !block.is_split() {
                    freed_size += block.size;
                    cur.remove_current();
                }
                cur.move_next();
            } else {
                break;
            }
        }
        freed_size
    }
}

#[derive(Clone)]
pub struct OpenClMemory {
    pool: Weak<RefCell<Pool>>,
    block: Block,
    buffer: Buffer,
}

impl OpenClMemory {
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl Drop for OpenClMemory {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            let mut pool = RefCell::borrow_mut(&pool);
            pool.free(&self.block);
        }
    }
}

#[derive(Clone)]
struct Block {
    base_buffer: Rc<Buffer>,
    cap: usize,
    size: usize,
    offset: usize,
    age: Counter,
}

impl Block {
    fn new(buffer: Buffer) -> Self {
        Block {
            base_buffer: Rc::new(buffer.clone()),
            cap: buffer.size(),
            size: buffer.size(),
            offset: 0,
            age: Counter::new(),
        }
    }

    fn buffer(&self) -> Buffer {
        if self.cap == self.size && self.offset == 0 {
            self.base_buffer.as_ref().clone()
        } else {
            self.base_buffer.sub_buffer::<f32>(self.size, self.offset)
        }
    }

    fn mem_eq(&self, other: &Block) -> bool {
        Rc::ptr_eq(&self.base_buffer, &other.base_buffer)
    }

    fn split(self, size: usize) -> Option<(Block, Block)> {
        if self.size >= size {
            Some((
                Block {
                    base_buffer: self.base_buffer.clone(),
                    cap: self.cap,
                    size,
                    offset: self.offset,
                    age: Counter::new(),
                },
                Block {
                    base_buffer: self.base_buffer.clone(),
                    cap: self.cap,
                    size: self.size - size,
                    offset: self.offset + size,
                    age: Counter::new(),
                },
            ))
        } else {
            None
        }
    }

    fn is_split(&self) -> bool {
        self.size != self.cap
    }
}

#[derive(Clone)]
struct Counter(Cell<usize>);

impl Counter {
    pub fn new() -> Self {
        Counter(Cell::new(0))
    }

    pub fn val(&self) -> usize {
        self.0.get()
    }

    pub fn set(&self, val: usize) {
        self.0.set(val);
    }

    pub fn up(&self) {
        self.set(self.val() + 1);
    }

    pub fn down(&self) {
        let val = self.val();
        if val >= 1 {
            self.set(val - 1);
        } else {
            self.reset();
        }
    }

    pub fn reset(&self) {
        self.set(0);
    }
}
