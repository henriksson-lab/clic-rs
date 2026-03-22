//! `clic-rs` — pure Rust GPU image processing library, equivalent to CLIc.
//!
//! # Quick start
//!
//! ```no_run
//! use clic_rs::{BackendManager, array::{push, pull}, tier1};
//!
//! let device = BackendManager::get().get_device("", "gpu").unwrap();
//! let src = push(&vec![1.0f32; 100], 10, 10, 1, &device).unwrap();
//! let dst = tier1::gaussian_blur(&device, &src, None, 1.0, 1.0, 0.0).unwrap();
//! let result: Vec<f32> = pull::<f32>(&dst).unwrap();
//! ```

pub mod error;
pub mod types;
pub mod utils;
pub mod cache;
pub mod device;
pub mod backend;
pub mod backend_manager;
pub mod array;
pub mod execution;
pub mod tier0;
pub mod tier1;
pub mod tier2;
pub mod tier3;
pub mod tier4;
pub mod tier5;
pub mod tier6;
pub mod tier7;
pub mod tier8;

pub use error::{CleError, Result};
pub use types::{DType, MType};
pub use array::{Array, ArrayPtr};
pub use device::DeviceArc;
pub use backend_manager::BackendManager;
