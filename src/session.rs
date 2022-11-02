use crate::error::Error;
use crate::ops::{Category, NaryOperator, Operator, VariadicOperator};
use crate::session::context::Context;
use crate::session::memory::MemoryError;
use crate::tensor::Tensor;
use crate::var::Var;
use itertools::Itertools;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

pub mod context;
pub mod device;
pub mod memory;
pub mod profiler;
pub mod reactor;

#[derive(Debug)]
pub struct Program<'a> {
    input: Vec<&'a Var>,
    output: Vec<&'a Var>,
    routines: Vec<Routine<'a>>,
    //routines_notfused: Vec<Routine<'a>>,
}

// collection of computations maybe called multiples times by program
#[derive(Debug)]
pub struct Routine<'a> {
    input: Vec<&'a Var>,
    output: &'a Var,
    stack: Vec<StackElement<'a>>,

    exec_time: Cell<Option<f64>>,
    mem_req: Cell<Option<usize>>,
}

#[derive(Clone, Debug)]
enum OwnedRef<'a, T> {
    Ref(&'a T),
    Owned(T),
}

impl<'a, T> AsRef<T> for OwnedRef<'a, T> {
    fn as_ref(&self) -> &T {
        match self {
            OwnedRef::Ref(r) => r,
            OwnedRef::Owned(v) => v,
        }
    }
}

#[derive(Debug)]
enum StackElement<'a> {
    Data(Var),
    Operator(OwnedRef<'a, Operator>),
    Operation(OwnedRef<'a, Operator>, Vec<StackElement<'a>>),
}


impl<'a> Program<'a> {
    pub fn compile<I1, I2>(x: I1, y: I2) -> Self
        where
            I1: IntoIterator<Item=&'a Var>,
            I2: IntoIterator<Item=&'a Var>,
    {
        let x = x.into_iter().collect_vec();
        let y = y
            .into_iter()
            .sorted_by(|v1, v2| Ord::cmp(&v1.t_order(), &v2.t_order()))
            .collect_vec();

        // [high]

        const THRES: usize = 5;
        let mut total_num_ops = 0;
        // PHASE 1: DETERMINE STATE_SETS
        // We do this to split whole computational graph into multiple ones.
        // Nodes whose values must be retained in the memory (or the context)

        let mut nodes = HashSet::<&Var>::new(); // (unique) vars
        let mut node_num_edges = HashMap::<&Var, usize>::new(); // number of dependencies each node has on other nodes.

        let mut stack = Vec::<&Var>::new();

        // populate nodes
        for k in y.iter() {
            stack.push(k);
            while !stack.is_empty() {
                let v = stack.pop().unwrap();

                if nodes.insert(v) && !x.contains(&v) {
                    if let Some(op) = v.op() {
                        stack.extend(op.input())
                    }
                }
            }
            stack.clear();
        }

        // populate node_num_edges
        for node in nodes {
            if let Some(op) = node.op() {
                for v in op.input() {
                    let count = node_num_edges.entry(v).or_insert(0);
                    *count += 1;
                }
            }
        }

        //states.extend(y.iter().cloned());

        // nodes with (num(edge) > 2), sorted by its topological order (lower order first)
        let state_candidates = node_num_edges
            .into_iter()
            .filter(|(v, num_edges)| {
                !(*num_edges < 2 || !v.is_op() || x.contains(v) || y.contains(v))
            })
            .map(|(v, _)| v)
            .chain(y.clone())
            .sorted_by(|v1, v2| Ord::cmp(&v1.t_order(), &v2.t_order()));

        // get vars with two or more dependencies.
        //let mut global_inputs = HashSet::new();
        let mut states = HashMap::<&Var, HashSet<&Var>>::new();

        for node in state_candidates {
            // do some *subgraph reduction*
            // We only store muldep nodes whose dependencies are complex enough (for memory efficiency and better latency)

            // find non-state + noncomplex

            // up... up.. to the root until met its origins.
            // why we do this? because we have to evaluate its 'worthiness' being retained in the memory.

            //let mut stack = Vec::new();
            stack.push(node);

            let mut inputs = HashSet::new();
            let mut num_ops = 0;
            let mut is_complex_op = false;

            while !stack.is_empty() {
                let v = stack.pop().unwrap();
                num_ops += 1;

                // this is the reason why nodes are accessed in topological order..
                // to make sure state_set.contains() is done in the right order.
                if v.is_data() || states.contains_key(v) || x.contains(&v) {
                    inputs.insert(v);
                }
                // interim
                else if let Some(op) = v.op() {
                    stack.extend(op.input().iter());

                    // if there is a complex op in the middle, it should be stored in the memory.
                    match op.cat() {
                        Category::Contract | Category::Reduce(_) | Category::Other => {
                            is_complex_op = true;
                        }
                        _ => {}
                    }
                } else {
                    panic!("uninitialized variable");
                }
            }
            // has complex op ? -> add to states
            // (num_input) * graph_rank * deps > some threshold -> add to states
            if y.contains(&node) || (is_complex_op || num_ops * inputs.len() > THRES) {
                total_num_ops += num_ops;
                states.insert(node, inputs);
            }
            stack.clear();
        }

        ////// PHASE 2: create sub-graphs divided by state_sets //////

        let mut program_inputs = HashSet::<&Var>::new();
        let mut routines = Vec::new();
        //let mut routines_notfused = Vec::new();


        // this loop can be done in parallel.
        for (node, inputs) in states {
            program_inputs.extend(inputs.iter());

            let routine = Routine::compile(inputs.clone(), node, true);
            //let routine_notfused = Routine::compile(inputs.into_iter(), node, false);

            routines.push(routine);
            //routines_notfused.push(routine_notfused);
        }

        //println!("total {:?} routines", routines.len());

        let input = program_inputs.into_iter().collect_vec();

        //println!("total ops: {:?}", total_num_ops);

        Program {
            input,
            output: y,
            routines,
            // routines_notfused,
        }
    }

    pub fn exec(&self, ctx: &mut Context) {
        let vf: HashMap<&Var, &Routine> = self
            .routines
            .iter()
            .map(|v| (v.output, v))
            .collect::<HashMap<&Var, &Routine>>();

        let mut lock = HashSet::<&Var>::new();

        let mut stack = Vec::<&Var>::new();
        stack.extend(&self.output);

        //let mut interm; // state && !output

        let mut ctx_temp = HashMap::<Var, Tensor>::new();
        let mut data_stack = Vec::new();

        while !stack.is_empty() {
            let v = stack.last().unwrap();

            if v.is_data() || ctx.data.contains_key(v) || ctx_temp.contains_key(v) {
                stack.pop();
            } else {
                let f = vf.get(v).unwrap();

                for k in f.input.iter() {
                    lock.insert(k);
                }

                let nr = f
                    .input
                    .iter()
                    .filter(|k| k.is_op() && !ctx.data.contains_key(k) && !ctx_temp.contains_key(k))
                    .cloned()
                    .collect::<Vec<&Var>>();

                // ready to be evaluated
                if nr.is_empty() {
                    let mut stack_ptr = 0;

                    loop {
                        match f.exec(ctx, &mut stack_ptr, &mut data_stack, &ctx_temp) {
                            Ok(out) => {
                                if self.output.contains(v) {
                                    ctx.data.insert((*v).clone(), out);
                                } else {
                                    ctx_temp.insert((*v).clone(), out);
                                }
                                break;
                            }

                            Err(e) => {
                                let Error::Memory(me) = e else {
                                    panic!("{:?}", e);
                                };

                                // only handle oom error
                                let MemoryError::OutOfMemory { mem_needed, mem_total } = me else {
                                    panic!("{:?}", me);
                                };

                                // await all tensors in ctx_temp
                                // tensor.synchronize().await

                                ////////////
                                let mem_start = ctx.memory.mem_used();
                                //let mut mem_freed = 0;

                                let mut c = vf
                                    .iter()
                                    .filter(|(v, _)| {
                                        // 당연히 recomputation이 가능해야 함
                                        v.is_op()
                                            // 이 조건이 없으면 무한 루프에 빠질 수 있음
                                            && !f.input.contains(v)
                                            // 당연하게도 메모리 안에 있어야 함
                                            && ctx_temp.contains_key(v)
                                            // 무한루프 방지용
                                            && !lock.contains(*v)
                                    })
                                    .collect_vec();

                                //let freeable_size = c.iter().map(|(v, _)| v.size()).sum::<usize>();

                                //println!("mem_available: {:?} MB / mem_needed: {:?} MB / freeable_size: {:?} MB",(ctx.memory.mem_total() - ctx.memory.mem_used()) * 4 / (1024 * 1024),mem_needed * 4 / (1024 * 1024) ,freeable_size * 4 / (1024 * 1024));

                                while mem_start - ctx.memory.mem_used() < mem_needed {
                                    //println!("freed! {:?} MB", mem_freed * 4 / (1024 * 1024));
                                    let mut tpm_min = 0.0;
                                    let mut min_entry = None;
                                    let mut min_entry_idx = 0;
                                    //println!("looking...");
                                    for (i, (v, r)) in c.iter().enumerate() {
                                        let tpm =
                                            (r.exec_time() + 1.0) / (r.output.size() as f64 + 1.0);

                                        //println!("{:?} / {:?}", tpm, tpm_min);

                                        if min_entry.is_none() || tpm_min > tpm {
                                            tpm_min = tpm;
                                            min_entry = Some(v);
                                            min_entry_idx = i;
                                        }
                                    }
                                    //println!("found!...");

                                    if let Some(v) = min_entry {
                                        //println!("{:?}", v);
                                        let tensor = ctx_temp.remove(v).unwrap();
                                        //println!("{:?}", Arc::strong_count(&tensor.data));

                                        //mem_freed += v.size();
                                        c.remove(min_entry_idx);
                                    } else {
                                        panic!("out of memory error");
                                    }
                                    //println!("next?!...");
                                }

                                // interim 중 가장 cost가 낮은 것 삭제

                                //////////////
                            }
                        }
                    }

                    for k in f.input.iter() {
                        lock.remove(k);
                    }

                    // if let Some(mem_cap) = ctx.memory.mem_cap() {
                    //     println!(
                    //         "mem_cap: {} MB / mem_used: {} MB",
                    //         mem_cap * 4 / (1024 * 1024),
                    //         ctx.memory.mem_used() * 4 / (1024 * 1024)
                    //     );
                    //
                    //     let mem_available = mem_cap - ctx.memory.mem_used();
                    //     let mem_required = f.mem_req();
                    //     // println!(
                    //     //     "mem_req: {} MB / mem_ava: {} MB",
                    //     //     f.mem_req() * 4 / (1024 * 1024),
                    //     //     mem_available * 4 / (1024 * 1024)
                    //     // );
                    //
                    //     // no memory left!!
                    //     if mem_required > mem_available {
                    //         // do dynamic checkpointing
                    //         //println!("dynamic gc!");
                    //         // iterate vf
                    //         let mem_needed = mem_required - mem_available;
                    //         let mut mem_freed = 0;
                    //
                    //         let mut c = vf
                    //             .iter()
                    //             .filter(|(v, _)| {
                    //                 // 당연히 recomputation이 가능해야 함
                    //                 v.is_op()
                    //                     // 이 조건이 없으면 무한 루프에 빠질 수 있음
                    //                     && !f.input.contains(v)
                    //                     // 당연하게도 메모리 안에 있어야 함
                    //                     && ctx_temp.contains_key(v)
                    //                     // 무한루프 방지용
                    //                     && !lock.contains(*v)
                    //             })
                    //             .collect_vec();
                    //
                    //         //let freeable_size = c.iter().map(|(v, _)| v.size()).sum::<usize>();
                    //
                    //         //println!("mem_available: {:?} MB / mem_needed: {:?} MB / freeable_size: {:?} MB",mem_available * 4 / (1024 * 1024),mem_needed * 4 / (1024 * 1024) ,freeable_size * 4 / (1024 * 1024));
                    //
                    //         while mem_freed < mem_needed {
                    //             //println!("freed! {:?} MB", mem_freed * 4 / (1024 * 1024));
                    //             let mut tpm_min = 0.0;
                    //             let mut min_entry = None;
                    //             let mut min_entry_idx = 0;
                    //             //println!("looking...");
                    //             for (i, (v, r)) in c.iter().enumerate() {
                    //                 let tpm = (r.exec_time() + 1.0) / (r.mem_req() as f64 + 1.0);
                    //
                    //                 //println!("{:?} / {:?}", tpm, tpm_min);
                    //
                    //                 if min_entry.is_none() || tpm_min > tpm {
                    //                     tpm_min = tpm;
                    //                     min_entry = Some(v);
                    //                     min_entry_idx = i;
                    //                 }
                    //             }
                    //             //println!("found!...");
                    //
                    //             if let Some(v) = min_entry {
                    //                 //println!("{:?}", v);
                    //                 let tensor = ctx_temp.remove(v).unwrap();
                    //                 //println!("{:?}", Arc::strong_count(&tensor.data));
                    //
                    //                 mem_freed += v.size();
                    //                 c.remove(min_entry_idx);
                    //             } else {
                    //                 panic!("out of memory error");
                    //                 break;
                    //             }
                    //             //println!("next?!...");
                    //         }
                    //
                    //         // interim 중 가장 cost가 낮은 것 삭제
                    //     }
                    // }
                    // if let Some(mem_cap) = ctx.memory.mem_cap() {
                    //     let mem_available = mem_cap - ctx.memory.mem_used();
                    //
                    //     if f.mem_req() > mem_available {
                    //         println!(
                    //             "  freed: {} MB / mem_ava: {} MB",
                    //             0 * 4 / (1024 * 1024),
                    //             mem_available * 4 / (1024 * 1024)
                    //         );
                    //     }
                    // }
                    // let out = f.exec(ctx, &ctx_temp).unwrap();
                    //
                    // if self.output.contains(v) {
                    //     ctx.data.insert((*v).clone(), out);
                    // } else {
                    //     ctx_temp.insert((*v).clone(), out);
                    // }
                    //
                    // for k in f.input.iter() {
                    //     lock.remove(k);
                    // }

                    // input 중 (op && !output) 인 것들의 counter 1 감소.
                    // 만약 counter 가 0이면 free
                    // for v in f.input.iter() {
                    //     if let Some(&count) = how_many_in_stack.get(v) {
                    //         if count <= 1 {
                    //             ctx_temp.remove(v);
                    //             how_many_in_stack.remove(v);
                    //         } else {
                    //             how_many_in_stack.insert(v, count - 1);
                    //         }
                    //     }
                    // }

                    stack.pop();
                } else {
                    // nr 중 (op && !output) 인 것들의 counter 1증가.

                    // for v in f
                    //     .input
                    //     .iter()
                    //     .filter(|k| k.is_op() && !self.output.contains(k))
                    // {
                    //     let count = how_many_in_stack.entry(v).or_insert(0);
                    //     *count += 1;
                    // }

                    stack.extend(nr);
                }
            }
        }
    }
    //
    // pub fn mem_req(&self, ctx: &mut Context) -> usize {
    //     let vf: HashMap<&Var, &Routine> = self
    //         .routines
    //         .iter()
    //         .map(|v| (v.output, v))
    //         .collect::<HashMap<&Var, &Routine>>();
    //
    //     let mut stack = Vec::<&Var>::new();
    //     stack.extend(&self.output);
    //
    //     while !stack.is_empty() {
    //         let v = stack.last().unwrap();
    //
    //         if v.is_data() || ctx.data.contains_key(v) {
    //             stack.pop();
    //         } else {
    //             let f = vf.get(v).unwrap();
    //
    //             let nr = f
    //                 .input
    //                 .iter()
    //                 .filter(|k| k.is_op() && !ctx.data.contains_key(k))
    //                 .cloned()
    //                 .collect::<Vec<&Var>>();
    //
    //             // ready to be evaluated
    //             if nr.is_empty() {
    //                 //let out = f.exec(ctx);
    //
    //                 //ctx.data.insert((*v).clone(), out);
    //
    //                 stack.pop();
    //             } else {
    //                 stack.extend(nr);
    //             }
    //         }
    //     }
    //
    //     0
    // }
}

// All operations are executed in the same stream.
impl<'a> Routine<'a> {
    pub fn compile<I>(x: I, y: &'a Var, op_fusion: bool) -> Self
        where
            I: IntoIterator<Item=&'a Var>,
    {
        let x = x.into_iter().collect_vec();

        // STEP1. build postfix tree
        let mut temp = Vec::<&'a Var>::new();
        let mut postfix = Vec::<StackElement<'a>>::new();

        temp.push(y);

        while !temp.is_empty() {
            let v = temp.pop().unwrap();

            if x.contains(&v) {
                postfix.push(StackElement::Data(v.clone()));
            } else if let Some(op) = v.op() {
                postfix.push(StackElement::Operator(OwnedRef::Ref(op.opr())));
                temp.extend(v.op().unwrap().input());
            }
        }

        // STEP2. build fused postfix tree
        let mut stack = Vec::new();

        while !postfix.is_empty() {
            let e = postfix.pop().unwrap();

            match e {
                // param -> working stack
                StackElement::Data(_) | StackElement::Operation(_, _) => stack.push(e),
                StackElement::Operator(opr) => {
                    let mut opr = opr;

                    let idx = stack.len() - opr.as_ref().arity();
                    // fuse operators with same memory access patterns + output type

                    let mut inner_stack = vec![];
                    let mut offset: i8 = 0;

                    for (i, p) in stack.drain(idx..).enumerate() {
                        match p {
                            StackElement::Data(_) => inner_stack.push(p),
                            StackElement::Operation(opr2, args) => {
                                // try operator fusion
                                if !op_fusion {
                                    inner_stack.push(StackElement::Operation(opr2, args));
                                } else if let Some(opr_fused) = Operator::fuse(
                                    opr.as_ref(),
                                    opr2.as_ref(),
                                    (i as i8 + offset) as usize,
                                ) {
                                    opr = OwnedRef::Owned(opr_fused);
                                    inner_stack.extend(args);
                                    offset += opr2.as_ref().arity() as i8 - 1;
                                } else {
                                    inner_stack.push(StackElement::Operation(opr2, args));
                                }
                            }
                            _ => {} // not reachable
                        }
                    }

                    postfix.push(StackElement::Operation(opr, inner_stack));
                }
            }
        }

        // always true
        assert_eq!(stack.len(), 1);

        let mut flat_stack = Vec::new();
        let mut next = vec![stack.pop().unwrap()]; //VecDeque::new();

        // flatten (no StackElement::Operation)
        while !next.is_empty() {
            let e = next.pop().unwrap();

            match e {
                StackElement::Data(_) => flat_stack.push(e),
                StackElement::Operation(opr, inner_stack) => {
                    flat_stack.push(StackElement::Operator(opr));
                    next.extend(inner_stack.into_iter());
                }
                _ => unimplemented!(),
            }
        }

        Routine {
            input: x,
            output: y,
            stack: flat_stack,
            mem_req: Cell::new(None),
            exec_time: Cell::new(None),
        }
    }

    fn exec(
        &self,
        ctx: &mut Context,
        stack_ptr: &mut usize,
        data_stack: &mut Vec<Tensor>,
        ctx_temp: &HashMap<Var, Tensor>,
    ) -> Result<Tensor, Error> {
        let now = Instant::now();

        //let mut stack = self.stack.iter().collect_vec();
        //let mut tensor_stack = Vec::new();

        //let mut program_ptr = 0;

        let stack_len = self.stack.len();

        while *stack_ptr < stack_len {
            let e = &self.stack[stack_len - *stack_ptr - 1];
            //println!("{:?}, {:?}, {:?}", stack_ptr, stack_len, data_stack.len());
            match e {
                StackElement::Data(v) => {
                    //println!("{:?}", v);
                    let data = v
                        .data()
                        .or_else(|| ctx.data.get(v).cloned())
                        .or_else(|| ctx_temp.get(v).cloned())
                        .unwrap();

                    data_stack.push(data);
                }

                StackElement::Operator(opr) => {
                    let opr = opr.as_ref();

                    //println!(" -- {:?}", opr.arity());

                    let idx = data_stack.len() - opr.arity();
                    //let input_tensors = data_stack.drain(idx..).collect_vec();

                    let input_tensors = &data_stack[idx..];

                    // assert!(input_tensors.iter().map(|v| v.desc()).zip(opr.input().iter()).all(|(a, b)| {
                    //     a == b
                    // }));

                    let output = opr.compute(input_tensors, ctx)?;
                    data_stack.truncate(idx);
                    data_stack.push(output);
                }
                _ => {}
            }

            *stack_ptr += 1;
        }

        let elapsed = now.elapsed().as_secs_f64();
        self.exec_time.set(Some(elapsed));

        assert_eq!(data_stack.len(), 1);

        Ok(data_stack.pop().unwrap())
    }

    pub fn exec_time(&self) -> f64 {
        self.exec_time.get().unwrap_or(0.0)
    }
    //
    // pub fn mem_req(&self) -> usize {
    //     if let Some(mem_req) = self.mem_req.get() {
    //         return mem_req;
    //     }
    //
    //     let mut stack = self.stack.iter().collect_vec();
    //     let mut mem_stack = Vec::new();
    //
    //     let mut mem = 0;
    //     let mut mem_peak = 0;
    //
    //     while !stack.is_empty() {
    //         let e = stack.pop().unwrap();
    //
    //         match e {
    //             StackElement::Data(_) => {
    //                 mem_stack.push(0);
    //             }
    //
    //             StackElement::Operator(opr) => {
    //                 let opr = opr.as_ref();
    //
    //                 let idx = mem_stack.len() - opr.arity();
    //                 let mem_used = opr.mem_req();
    //                 let mem_freed = mem_stack.drain(idx..).sum::<usize>();
    //
    //                 mem_stack.push(mem_used);
    //
    //                 mem += mem_used;
    //                 if mem > mem_peak {
    //                     mem_peak = mem;
    //                 }
    //                 mem -= mem_freed;
    //             }
    //             _ => {}
    //         }
    //     }
    //     self.mem_req.set(Some(mem_peak));
    //     mem_peak
    // }
}
