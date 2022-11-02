use std::{
    collections::HashMap,
    future::Future,
    mem,
    pin::Pin,
    sync::{
        mpsc::{channel, Sender},
        Arc, Condvar, Mutex,
    },
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

enum TaskState {
    Ready,
    NotReady(Waker),
    Finished,
}

pub struct Reactor {
    dispatcher: Sender<Event>,
    handle: Option<JoinHandle<()>>,
    tasks: HashMap<usize, TaskState>,
}

#[derive(Debug)]
enum Event {
    Close,
    Timeout(u64, usize),
}

impl Reactor {
    pub fn new() -> Arc<Mutex<Box<Self>>> {
        let (tx, rx) = channel::<Event>();
        let reactor = Arc::new(Mutex::new(Box::new(Reactor {
            dispatcher: tx,
            handle: None,
            tasks: HashMap::new(),
        })));

        let reactor_clone = Arc::downgrade(&reactor);
        let (tx2, rx2) = channel::<Event>();


        let compiler_handle = thread::spawn(move || {
            for event in rx {
                match event {
                    Event::Close => {
                        tx2.send(Event::Close).unwrap();
                        break
                    },
                    Event::Timeout(duration, id) => {

                        // do graph compilation

                        // finished compilation
                        tx2.send(Event::Timeout(duration, id)).unwrap();
                    }
                }
            }
        });

        let cuda_handle = thread::spawn(move || {
            for event in rx2 {
                //let reactor = reactor_clone.clone();
                match event {
                    Event::Close => break,
                    Event::Timeout(duration, id) => {

                        // triage operations to corresponding cuda streams
                        // for now, use just single stream





                    }
                }
            }

        });

        // cuda event poller



        reactor.lock().map(|mut r| r.handle = Some(cuda_handle)).unwrap();
        reactor
    }

    fn wake(&mut self, id: usize) {
        let state = self.tasks.get_mut(&id).unwrap();
        match mem::replace(state, TaskState::Ready) {
            TaskState::NotReady(waker) => waker.wake(),
            TaskState::Finished => panic!("Called 'wake' twice on task: {}", id),
            _ => unreachable!(),
        }
    }

    fn register(&mut self, duration: u64, waker: Waker, id: usize) {
        if self.tasks.insert(id, TaskState::NotReady(waker)).is_some() {
            panic!("Tried to insert a task with id: '{}', twice!", id);
        }
        self.dispatcher.send(Event::Timeout(duration, id)).unwrap();
    }

    fn is_ready(&self, id: usize) -> bool {
        self.tasks
            .get(&id)
            .map(|state| matches!(state, TaskState::Ready))
            .unwrap_or(false)
    }
}

impl Drop for Reactor {
    fn drop(&mut self) {
        self.dispatcher.send(Event::Close).unwrap();
        self.handle.take().map(|h| h.join().unwrap()).unwrap();
    }
}
