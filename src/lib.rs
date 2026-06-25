//! `clic-rs` — pure Rust GPU image processing library, equivalent to CLIc.
//!
//! # Quick start
//!
//! ```no_run
//! use clic_rs::{BackendManager, array::Array, tier1};
//!
//! let device = BackendManager::get_instance().get_device("", "gpu").unwrap();
//! let src = clic_rs::array::Array::create_with_data(10, 10, 1, clic_rs::utils::shape_to_dimension(10, 10, 1), clic_rs::types::MType::Buffer, &vec![1.0f32; 100], &device).unwrap();
//! let dst = tier1::gaussian_blur(&device, &src, None, 1.0, 1.0, 0.0).unwrap();
//! let mut result = vec![0.0_f32; dst.lock().unwrap().size()];
//! dst.lock().unwrap().read_to(&mut result).unwrap();
//! ```

pub mod array;
pub mod backend;
pub mod backend_manager;
pub mod cache;
pub mod device;
pub mod error;
pub mod execution;
pub mod fft;
pub mod slicing;
pub mod statistics;
pub mod tier0;
pub mod tier1;
pub mod tier2;
pub mod tier3;
pub mod tier4;
pub mod tier5;
pub mod tier6;
pub mod tier7;
pub mod tier8;
pub mod transform;
pub mod translator;
pub mod types;
pub mod utils;

pub use array::{Array, ArrayPtr};
pub use backend_manager::BackendManager;
pub use device::DeviceArc;
pub use error::{CleError, Result};
pub use types::{DType, MType};
