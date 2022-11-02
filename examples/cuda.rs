// #[macro_use]
// extern crate rustacuda;
//
// use rustacuda::prelude::*;
// use rustacuda::memory::DeviceBox;
// use std::error::Error;
// use std::ffi::CString;
// use std::fs;
//
// use std::process::Command;
//
// use {
//     futures::{
//         future::{BoxFuture, FutureExt},
//         task::{waker_ref, ArcWake},
//         join,
//     },
//     std::{
//         thread,
//         pin::Pin,
//         future::Future,
//         sync::mpsc::{sync_channel, Receiver, SyncSender},
//         sync::{Arc, Mutex},
//         task::{Context, Poll, Waker},
//         time::Duration,
//     },
//     // The timer we wrote in the previous section:
// };
//
//
// pub struct TimerFuture {
//     shared_state: Arc<Mutex<SharedState>>,
// }
//
// /// Shared state between the future and the waiting thread
// struct SharedState {
//     /// Whether or not the sleep time has elapsed
//     completed: bool,
//
//     /// The waker for the task that `TimerFuture` is running on.
//     /// The thread can use this after setting `completed = true` to tell
//     /// `TimerFuture`'s task to wake up, see that `completed = true`, and
//     /// move forward.
//     waker: Option<Waker>,
// }
//
// impl TimerFuture {
//     /// Create a new `TimerFuture` which will complete after the provided
//     /// timeout.
//     pub fn new(duration: Duration) -> Self {
//         let shared_state = Arc::new(Mutex::new(SharedState {
//             completed: false,
//             waker: None,
//         }));
//
//         // Spawn the new thread
//         let thread_shared_state = shared_state.clone();
//         thread::spawn(move || {
//             thread::sleep(duration);
//             let mut shared_state = thread_shared_state.lock().unwrap();
//             // Signal that the timer has completed and wake up the last
//             // task on which the future was polled, if one exists.
//             shared_state.completed = true;
//             if let Some(waker) = shared_state.waker.take() {
//                 waker.wake()
//             }
//         });
//
//         TimerFuture { shared_state }
//     }
// }
//
// impl Future for TimerFuture {
//     type Output = ();
//     fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
//         // Look at the shared state to see if the timer has already completed.
//         let mut shared_state = self.shared_state.lock().unwrap();
//         if shared_state.completed {
//             Poll::Ready(())
//         } else {
//             // Set waker so that the thread can wake up the current task
//             // when the timer has completed, ensuring that the future is polled
//             // again and sees that `completed = true`.
//             //
//             // It's tempting to do this once rather than repeatedly cloning
//             // the waker each time. However, the `TimerFuture` can move between
//             // tasks on the executor, which could cause a stale waker pointing
//             // to the wrong task, preventing `TimerFuture` from waking up
//             // correctly.
//             //
//             // N.B. it's possible to check for this using the `Waker::will_wake`
//             // function, but we omit that here to keep things simple.
//             shared_state.waker = Some(cx.waker().clone());
//             Poll::Pending
//         }
//     }
// }
//
// /// Task executor that receives tasks off of a channel and runs them.
// struct Executor {
//     ready_queue: Receiver<Arc<Task>>,
// }
//
// /// `Spawner` spawns new futures onto the task channel.
// #[derive(Clone)]
// struct Spawner {
//     task_sender: SyncSender<Arc<Task>>,
// }
//
// /// A future that can reschedule itself to be polled by an `Executor`.
// struct Task {
//     /// In-progress future that should be pushed to completion.
//     ///
//     /// The `Mutex` is not necessary for correctness, since we only have
//     /// one thread executing tasks at once. However, Rust isn't smart
//     /// enough to know that `future` is only mutated from one thread,
//     /// so we need to use the `Mutex` to prove thread-safety. A production
//     /// executor would not need this, and could use `UnsafeCell` instead.
//     future: Mutex<Option<BoxFuture<'static, ()>>>,
//
//     /// Handle to place the task itself back onto the task queue.
//     task_sender: SyncSender<Arc<Task>>,
// }
//
// fn new_executor_and_spawner() -> (Executor, Spawner) {
//     // Maximum number of tasks to allow queueing in the channel at once.
//     // This is just to make `sync_channel` happy, and wouldn't be present in
//     // a real executor.
//     const MAX_QUEUED_TASKS: usize = 10_000;
//     let (task_sender, ready_queue) = sync_channel(MAX_QUEUED_TASKS);
//     (Executor { ready_queue }, Spawner { task_sender })
// }
//
//
// impl Spawner {
//     fn spawn(&self, future: impl Future<Output=()> + 'static + Send) {
//         let future = future.boxed();
//         let task = Arc::new(Task {
//             future: Mutex::new(Some(future)),
//             task_sender: self.task_sender.clone(),
//         });
//         self.task_sender.send(task).expect("too many tasks queued");
//     }
// }
//
// impl ArcWake for Task {
//     fn wake_by_ref(arc_self: &Arc<Self>) {
//         // Implement `wake` by sending this task back onto the task channel
//         // so that it will be polled again by the executor.
//         let cloned = arc_self.clone();
//         arc_self
//             .task_sender
//             .send(cloned)
//             .expect("too many tasks queued");
//     }
// }
//
// impl Executor {
//     fn run(&self) {
//         while let Ok(task) = self.ready_queue.recv() {
//             // Take the future, and if it has not yet completed (is still Some),
//             // poll it in an attempt to complete it.
//             let mut future_slot = task.future.lock().unwrap();
//             if let Some(mut future) = future_slot.take() {
//                 // Create a `LocalWaker` from the task itself
//                 let waker = waker_ref(&task);
//                 let context = &mut Context::from_waker(&*waker);
//                 // `BoxFuture<T>` is a type alias for
//                 // `Pin<Box<dyn Future<Output = T> + Send + 'static>>`.
//                 // We can get a `Pin<&mut dyn Future + Send + 'static>`
//                 // from it by calling the `Pin::as_mut` method.
//                 if future.as_mut().poll(context).is_pending() {
//                     // We're not done processing the future, so put it
//                     // back in its task to be run again in the future.
//                     *future_slot = Some(future);
//                 }
//             }
//         }
//     }
// }
//
//
// fn main() -> Result<(), Box<dyn Error>> {
//     let (executor, spawner) = new_executor_and_spawner();
//
//     // Spawn a task to print before and after waiting on a timer.
//     spawner.spawn(async {
//         println!("howdy!");
//         // Wait for our timer future to complete after two seconds.
//         let task1 = TimerFuture::new(Duration::new(2, 0));
//         let task2 = TimerFuture::new(Duration::new(3, 0));
//
//         join!(task1, task2);
//         println!("done!");
//     });
//
//     // Drop the spawner so that our executor knows it is finished and won't
//     // receive more incoming tasks to run.
//     drop(spawner);
//
//     // Run the executor until the task queue is empty.
//     // This will print "howdy!", pause, and then print "done!".
//     executor.run();
//
//     let src = compile_cuda(r#"
// extern "C"  __global__ void sum(const float* x, const float* y, float* out, int count) {
//     for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
//         out[i] = x[i] + y[i];
//     }
// }
// "#);
//
//     // Initialize the CUDA API
//     rustacuda::init(CudaFlags::empty())?;
//
//     // Get the first backend
//     let device = Device::get_device(0)?;
//
//     // Create a context associated to this backend
//     let context = rustacuda::context::Context::create_and_push(
//         ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
//
//     // Load the module containing the function we want to call
//     let module = Module::load_from_string(&src)?;
//
//     // Create a stream to submit work to
//     let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
//
//     // Create buffers for data
//     let mut in_x = DeviceBuffer::from_slice(&[1.0f32; 10])?;
//     let mut in_y = DeviceBuffer::from_slice(&[2.0f32; 10])?;
//     let mut out_1 = DeviceBuffer::from_slice(&[0.0f32; 10])?;
//     let mut out_2 = DeviceBuffer::from_slice(&[0.0f32; 10])?;
//
//     // This kernel adds each element in `in_x` and `in_y` and writes the result into `out`.
//     unsafe {
//         // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
//         let result = launch!(module.sum<<<1, 1, 0, stream>>>(
//             in_x.as_device_ptr(),
//             in_y.as_device_ptr(),
//             out_1.as_device_ptr(),
//             out_1.len()
//         ));
//         result?;
//
//         // Launch the kernel again using the `function` form:
//         let function_name = CString::new("sum")?;
//         let sum = module.get_function(&function_name)?;
//         // Launch with 1x1x1 (1) blocks of 10x1x1 (10) threads, to show that you can use tuples to
//         // configure grid and block size.
//         let result = launch!(sum<<<(1, 1, 1), (10, 1, 1), 0, stream>>>(
//             in_x.as_device_ptr(),
//             in_y.as_device_ptr(),
//             out_2.as_device_ptr(),
//             out_2.len()
//         ));
//         result?;
//     }
//
//     // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
//     stream.synchronize()?;
//
//     // Copy the results back to host memory
//     let mut out_host = [0.0f32; 20];
//     out_1.copy_to(&mut out_host[0..10])?;
//     out_2.copy_to(&mut out_host[10..20])?;
//
//     for x in out_host.iter() {
//         assert_eq!(3.0 as u32, *x as u32);
//     }
//
//     println!("Launched kernel successfully.");
//     Ok(())
// }
//
//
// fn compile_cuda(code: &str) -> CString {
//     // step 1. save file
//
//     // step 2. call nvcc
//
//     let filename = "asd123";
//     fs::create_dir_all("./.cache/kernels/").unwrap();
//
//     let input_path = format!("./.cache/kernels/{filename}.cu");
//     let output_path = format!("./.cache/kernels/{filename}.ptx");
//
//
//     // step 3. load string
//     fs::write(&input_path, code).expect("Unable to write file");
//
//     println!("start building...");
//
//     let output = Command::new("nvcc")
//         .arg(&input_path)
//         .arg(format!("-o={output_path}"))
//         .arg("-c")
//         .arg("--ptx")
//         //.arg("-cudart=shared")
//         //.arg("-gencode")
//         //.arg("arch=compute_61,code=sm_61")
//         .output()
//         .expect("failed to execute process");
//
//
//     let aa = fs::read_to_string(output_path).expect("unable to read file!");
//     CString::new(aa).unwrap()
//     //CString::new("asdasd").unwrap()
// }
fn main() {}