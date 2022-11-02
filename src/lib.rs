#![feature(vec_into_raw_parts)]
#![feature(is_sorted)]
#![feature(new_uninit)]
#![feature(slice_take)]
#![feature(array_zip)]
#![feature(bool_to_option)]
#![feature(linked_list_cursors)]
#![feature(linked_list_remove)]
#![feature(drain_filter)]
#![feature(let_else)]

pub mod dataset;
pub mod error;
pub mod layers;
pub mod models;
pub mod ops;
pub mod optim;
pub mod session;
pub mod shape;
pub mod tensor;
pub mod var;
pub mod backend;

extern crate core;

