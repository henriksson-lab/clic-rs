use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::backend::{Backend, OpenCLBackend};
use crate::device::{enumerate_opencl_devices, DeviceArc};
use crate::error::{CleError, Result};

/// Global backend manager — holds the active backend (OpenCL by default).
pub struct BackendManager {
    backend: Box<dyn Backend>,
}

static INSTANCE: OnceLock<Mutex<BackendManager>> = OnceLock::new();

impl BackendManager {
    /// Access the global singleton.
    pub fn get() -> MutexGuard<'static, BackendManager> {
        INSTANCE
            .get_or_init(|| Mutex::new(BackendManager { backend: Box::new(OpenCLBackend) }))
            .lock()
            .unwrap()
    }

    /// Access the active backend.
    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Replace the active backend ("opencl" is the only supported option currently).
    pub fn set_backend(&mut self, name: &str) -> Result<()> {
        match name.to_lowercase().as_str() {
            "opencl" => {
                self.backend = Box::new(OpenCLBackend);
                Ok(())
            }
            other => Err(CleError::Other(format!("Unknown backend: {}", other))),
        }
    }

    /// Enumerate available devices, filtered by `device_type` ("gpu", "cpu", "all").
    pub fn get_devices(&self, device_type: &str) -> Result<Vec<DeviceArc>> {
        enumerate_opencl_devices(device_type)
    }

    /// Return the best available device (GPU preferred, any type as fallback).
    pub fn get_device(&self, name: &str, device_type: &str) -> Result<DeviceArc> {
        let dtype = device_type.to_lowercase();
        let mut devices = enumerate_opencl_devices(&dtype)?;
        if devices.is_empty() {
            devices = enumerate_opencl_devices("all")?;
        }
        if devices.is_empty() {
            return Err(CleError::NoDevicesFound);
        }
        if name.is_empty() {
            return Ok(devices.pop().unwrap());
        }
        let lower = name.to_lowercase();
        devices
            .into_iter()
            .find(|d| d.name().to_lowercase().contains(&lower))
            .or_else(|| Some(enumerate_opencl_devices("all").ok()?.pop()?))
            .ok_or(CleError::NoDevicesFound)
    }
}
