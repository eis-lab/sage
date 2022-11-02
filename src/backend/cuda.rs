

use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use std::error::Error;
use std::ffi::CString;
use std::fs;

use std::process::Command;

thread_local!(static CUDA_INIT:bool = rustacuda::init(rustacuda::CudaFlags::empty()).is_ok());

pub struct Device {
    cuda_device: rustacuda::device::Device,
}

impl Device {
    pub fn list() -> Vec<Device> {
        let devices = rustacuda::device::Device::devices().unwrap();

        devices.map(|d| Device {
            cuda_device: d.unwrap()
        }).collect()
    }

    pub fn compile_ptx(code: &str) -> CString {
        // step 1. save file

        // step 2. call nvcc

        let filename = "asd123";
        fs::create_dir_all("./.cache/kernels/").unwrap();

        let input_path = format!("./.cache/kernels/{filename}.cu");
        let output_path = format!("./.cache/kernels/{filename}.ptx");


        // step 3. load string
        fs::write(&input_path, code).expect("Unable to write file");

        println!("start building...");

        let output = Command::new("nvcc")
            .arg(&input_path)
            .arg(format!("-o={output_path}"))
            .arg("-c")
            .arg("--ptx")
            //.arg("-cudart=shared")
            //.arg("-gencode")
            //.arg("arch=compute_61,code=sm_61")
            .output()
            .expect("failed to execute process");


        let aa = fs::read_to_string(output_path).expect("unable to read file!");
        CString::new(aa).unwrap()
    }

}

pub struct Stream {

}

pub struct Context {

}

/**

context::with(async |&mut c| {

    gg.eval(c).await?;

}


**/


// goes to the tensor..
pub struct Buffer {}

// backend::Cuda::list();
// backend::Cuda::new(0);

// backend::OpenCl::list();
// backend::OpenCl::new();

// let mut ctx = backend.create_context();

// Tensor::new().to(&mut ctx);
//
