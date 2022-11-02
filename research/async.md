# Sage async


- Latency hiding in graph optimization & jit compilation

- Front-dep nodes can benefit, back-dep nodes cannto benefit....??? 
    - front-deps:
        1. find `queued_vars` and only insert remaining subgraphs to the corresponding stream..

    - back-deps:
        1. find `queued_vars` and find itself -> lucky!!
        2. find `queued_vars` and only insert remaining subgraphs to the corresponding stream.. -> unlucky!!

    - no-deps:
        1. create new stream and push


### Executor design

Device
    - streams [1, 2, 3, 4, ..., n]


Do we need context?
    - memory optimizations...
    - incremental computation



1. var.eval(ctx)
2. spawn(graph evaluator + JIT compiler)     [single FIFO executor]
    -> add queued_list (routine, stream)
3. done? -> spawn(CUDA exec)                 * separate executor [each thread owns its CUDA stream]
    -> if input data is in the queud_list with different streams,
    -> put stream synchronize commands 


* Different design with no stream per thread


```rust
CudaExecutor {
    queue //  [kernel1, event_dispatch1, kernel2, kernel3, ...]
    eventpool
}

impl CudaExecutor {

    // stream scheduler


    pub fn pick_and_run(self) {

        // 
        let work = self.queue.pop()

        // select stream 

        // events

        if let Some(stream) = self.get_dependent_streams(&work) {
            // wait events 
            

        }


    }
}
```



```rust


struct Reactor {

}


impl Reactor {


    pub fn new() {

        
        // two threads: one for compilation, one for cuda execution





    }
}



```
