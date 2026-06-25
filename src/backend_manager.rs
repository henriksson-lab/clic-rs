use std::sync::{OnceLock, RwLock, RwLockReadGuard};

use opencl3::platform::get_platforms;

use crate::backend::{Backend, OpenCLBackend};
use crate::device::{enumerate_opencl_devices, DeviceArc};
use crate::error::{CleError, Result};

#[path = "cuda_backend.rs"]
pub mod cuda_backend;
#[path = "cuda_device.rs"]
pub mod cuda_device;

/// Global backend manager — holds the active backend (OpenCL by default).
pub struct BackendManager {
    backend: Box<dyn Backend>,
}

static INSTANCE: OnceLock<RwLock<BackendManager>> = OnceLock::new();
static CUDA_ERROR: RwLock<String> = RwLock::new(String::new());
static OPENCL_ERROR: RwLock<String> = RwLock::new(String::new());

impl BackendManager {
    fn new() -> Self {
        Self {
            backend: Box::new(OpenCLBackend::new()),
        }
    }

    /// Whether CUDA is available. CUDA is not compiled into this Rust backend.
    pub fn cuda_enabled() -> bool {
        let compiled_with_cuda = false;
        let mut device_count = 0usize;
        let mut error_message = String::new();

        if compiled_with_cuda {
            // Placeholder for a future CUDA backend: mirror CLIc's two-stage
            // check of driver initialization followed by device enumeration.
            device_count = 0;
        } else {
            error_message = "CUDA not compiled into this build (USE_CUDA=OFF)".to_string();
        }

        if compiled_with_cuda && device_count == 0 {
            error_message = "No CUDA devices found".to_string();
        }

        if error_message.is_empty() {
            CUDA_ERROR.write().unwrap().clear();
            true
        } else {
            *CUDA_ERROR.write().unwrap() = error_message;
            false
        }
    }

    /// Whether at least one OpenCL platform is available.
    pub fn opencl_enabled() -> bool {
        match get_platforms() {
            Ok(platforms) if !platforms.is_empty() => {
                OPENCL_ERROR.write().unwrap().clear();
                true
            }
            Ok(_) => {
                *OPENCL_ERROR.write().unwrap() = "No OpenCL platforms found".to_string();
                false
            }
            Err(err) => {
                *OPENCL_ERROR.write().unwrap() =
                    format!("clGetPlatformIDs failed with error code {:?}", err);
                false
            }
        }
    }

    /// Return available backend names. Mirrors CLIc's `getBackendsList()`.
    pub fn get_backends_list() -> Vec<String> {
        let mut backends = Vec::new();
        if Self::cuda_enabled() {
            backends.push("cuda".to_string());
        }
        if Self::opencl_enabled() {
            backends.push("opencl".to_string());
        }
        backends
    }

    /// Canonical singleton alias for CLIc's `getInstance()`.
    pub fn get_instance() -> RwLockReadGuard<'static, BackendManager> {
        INSTANCE
            .get_or_init(|| RwLock::new(BackendManager::new()))
            .read()
            .unwrap()
    }

    /// Replace the active backend ("opencl" is the only supported option currently).
    pub fn set_backend(&mut self, backend: &str) -> Result<()> {
        let backend_map = [("opencl", "opencl")];

        let Some((backend_name, backend_type)) =
            backend_map.iter().find(|(name, _)| *name == backend)
        else {
            let mut list = String::new();
            for (name, _) in backend_map {
                if !list.is_empty() {
                    list += ", ";
                }
                list += name;
            }
            return Err(CleError::Other(format!(
                "Unknown backend '{}'. This build supports: {}",
                backend,
                if list.is_empty() { "none" } else { &list }
            )));
        };

        let is_enabled = if *backend_type == "cuda" {
            Self::cuda_enabled()
        } else {
            Self::opencl_enabled()
        };
        if !is_enabled {
            let error_reason = if *backend_type == "cuda" {
                Self::get_cuda_error()
            } else {
                Self::get_opencl_error()
            };
            return Err(CleError::Other(format!(
                "Backend '{}' is not available: {}",
                backend_name, error_reason
            )));
        }

        if *backend_type == "opencl" {
            self.backend = Box::new(OpenCLBackend::new());
        }
        Ok(())
    }

    /// Last CUDA availability error. Mirrors CLIc's `getCudaError()`.
    pub fn get_cuda_error() -> String {
        CUDA_ERROR.read().unwrap().clone()
    }

    /// Last OpenCL availability error. Mirrors CLIc's `getOpenCLError()`.
    pub fn get_opencl_error() -> String {
        OPENCL_ERROR.read().unwrap().clone()
    }

    /// Canonical alias for CLIc's `getBackend()`.
    pub fn get_backend(&self) -> &dyn Backend {
        self.backend()
    }

    /// Access the active backend.
    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Return the best available device (GPU preferred, any type as fallback).
    pub fn get_device(&self, name: &str, device_type: &str) -> Result<DeviceArc> {
        let devices_all = enumerate_opencl_devices("all")?;
        if devices_all.is_empty() {
            eprintln!("Warning: Fail to find any OpenCL compatible devices.");
            return Err(CleError::NoDevicesFound);
        }
        let devices = enumerate_opencl_devices(device_type)?;
        if devices.is_empty() {
            return Ok(devices_all.last().unwrap().clone());
        }
        if name.is_empty() {
            return Ok(devices.last().unwrap().clone());
        }
        let lower = name.to_lowercase();
        Ok(devices
            .iter()
            .find(|device| device.get_name().to_lowercase().contains(&lower))
            .unwrap_or_else(|| devices.last().unwrap())
            .clone())
    }
}

impl Drop for BackendManager {
    fn drop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_availability_reports_compile_time_gap() {
        assert!(!BackendManager::cuda_enabled());
        assert_eq!(
            BackendManager::get_cuda_error(),
            "CUDA not compiled into this build (USE_CUDA=OFF)"
        );
    }

    #[test]
    fn backends_list_contains_only_available_backends() {
        let backends = BackendManager::get_backends_list();
        assert!(!backends.iter().any(|backend| backend == "cuda"));
        assert!(backends.iter().all(|backend| backend == "opencl"));
    }
}
