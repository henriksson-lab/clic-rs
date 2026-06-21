use std::sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::backend::{Backend, OpenCLBackend};
use crate::device::{enumerate_opencl_devices, DeviceArc};
use crate::error::{CleError, Result};

/// Global backend manager — holds the active backend (OpenCL by default).
pub struct BackendManager {
    backend: Box<dyn Backend>,
}

static INSTANCE: OnceLock<RwLock<BackendManager>> = OnceLock::new();

impl BackendManager {
    /// Access the global singleton (read-only — sufficient for kernel dispatch).
    pub fn get() -> RwLockReadGuard<'static, BackendManager> {
        INSTANCE
            .get_or_init(|| {
                RwLock::new(BackendManager {
                    backend: Box::new(OpenCLBackend),
                })
            })
            .read()
            .unwrap()
    }

    /// Canonical singleton alias for CLIc's `getInstance()`.
    pub fn get_instance() -> RwLockReadGuard<'static, BackendManager> {
        Self::get()
    }

    /// Access the global singleton with write permission (for `set_backend`).
    pub fn get_mut() -> RwLockWriteGuard<'static, BackendManager> {
        INSTANCE
            .get_or_init(|| {
                RwLock::new(BackendManager {
                    backend: Box::new(OpenCLBackend),
                })
            })
            .write()
            .unwrap()
    }

    /// Access the active backend.
    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Canonical alias for CLIc's `getBackend()`.
    pub fn get_backend(&self) -> &dyn Backend {
        self.backend()
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

    /// Return available device names for a device type. Mirrors CLIc's
    /// `getDevicesList()`.
    pub fn get_devices_list(&self, device_type: &str) -> Result<Vec<String>> {
        Ok(self
            .get_devices(device_type)?
            .into_iter()
            .map(|device| device.name().to_string())
            .collect())
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
            .or_else(|| enumerate_opencl_devices("all").ok()?.pop())
            .ok_or(CleError::NoDevicesFound)
    }

    /// Return a device by index and type. Mirrors CLIc's `getDeviceFromIndex()`.
    pub fn get_device_from_index(&self, index: usize, device_type: &str) -> Result<DeviceArc> {
        let devices = self.get_devices(device_type)?;
        devices
            .into_iter()
            .nth(index)
            .ok_or(CleError::NoDevicesFound)
    }
}
