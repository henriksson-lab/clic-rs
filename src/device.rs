use std::sync::{Arc, Mutex};

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{Device as ClDevice, CL_DEVICE_TYPE_ALL};
use opencl3::platform::get_platforms;
use opencl3::program::Program;

use crate::cache::new_shared_program_cache;
use crate::cache::{DiskCache, SharedProgramCache};
use crate::error::{CleError, Result};

/// Abstraction over a GPU device.
pub trait Device: Send + Sync {
    fn name(&self) -> &str;
    fn device_type(&self) -> &str;
    fn support_image(&self) -> bool;
    fn max_buffer_size(&self) -> usize;
    fn max_work_group_size(&self) -> usize;
    fn finish(&self);
    fn get_program_from_cache(&self, key: &str) -> Option<Arc<Program>>;
    fn add_program_to_cache(&self, key: String, program: Arc<Program>);
    fn device_hash(&self) -> String;
}

pub type DeviceArc = Arc<dyn Device>;

// ── OpenCL device ────────────────────────────────────────────────────────────

pub struct OpenCLDevice {
    pub(crate) ocl_device: ClDevice,
    pub(crate) context: Arc<Context>,
    pub(crate) queue: Arc<CommandQueue>,
    program_cache: SharedProgramCache,
    name: String,
    dtype: String,
    image_support: bool,
    max_buffer_size: usize,
    max_work_group_size: usize,
    device_hash: String,
}

// opencl3 types are Send + Sync (they explicitly implement it)
unsafe impl Send for OpenCLDevice {}
unsafe impl Sync for OpenCLDevice {}

impl OpenCLDevice {
    pub fn new(ocl_device: ClDevice, context: Arc<Context>, queue: Arc<CommandQueue>) -> Result<Self> {
        let name = ocl_device.name().unwrap_or_default();
        let dtype = match ocl_device.dev_type().unwrap_or(0) {
            opencl3::device::CL_DEVICE_TYPE_GPU => "gpu".to_string(),
            opencl3::device::CL_DEVICE_TYPE_CPU => "cpu".to_string(),
            _ => "other".to_string(),
        };
        let image_support = ocl_device.image_support().unwrap_or(false);
        let max_buffer_size = ocl_device.max_mem_alloc_size().unwrap_or(0) as usize;
        let max_work_group_size = ocl_device.max_work_group_size().unwrap_or(256);

        let driver = ocl_device.driver_version().unwrap_or_default();
        let device_hash = DiskCache::hash(&format!("{}:{}", name, driver));

        Ok(OpenCLDevice {
            ocl_device,
            context,
            queue,
            program_cache: new_shared_program_cache(),
            name,
            dtype,
            image_support,
            max_buffer_size,
            max_work_group_size,
            device_hash,
        })
    }
}

impl Device for OpenCLDevice {
    fn name(&self) -> &str { &self.name }
    fn device_type(&self) -> &str { &self.dtype }
    fn support_image(&self) -> bool { self.image_support }
    fn max_buffer_size(&self) -> usize { self.max_buffer_size }
    fn max_work_group_size(&self) -> usize { self.max_work_group_size }
    fn device_hash(&self) -> String { self.device_hash.clone() }
    fn finish(&self) { let _ = self.queue.finish(); }

    fn get_program_from_cache(&self, key: &str) -> Option<Arc<Program>> {
        self.program_cache.lock().unwrap().get(key)
    }
    fn add_program_to_cache(&self, key: String, program: Arc<Program>) {
        self.program_cache.lock().unwrap().put(key, program);
    }
}

// ── Enumerate devices ────────────────────────────────────────────────────────

pub fn enumerate_opencl_devices(device_type: &str) -> Result<Vec<DeviceArc>> {
    let platforms = get_platforms().map_err(|e| CleError::OpenCL(format!("{:?}", e)))?;
    let mut devices: Vec<DeviceArc> = Vec::new();

    for platform in platforms {
        let device_ids = platform.get_devices(CL_DEVICE_TYPE_ALL).unwrap_or_default();

        for &id in &device_ids {
            let cl_dev = ClDevice::new(id);

            // Filter by type
            if device_type != "all" {
                let dev_type = match cl_dev.dev_type().unwrap_or(0) {
                    opencl3::device::CL_DEVICE_TYPE_GPU => "gpu",
                    opencl3::device::CL_DEVICE_TYPE_CPU => "cpu",
                    _ => "other",
                };
                if dev_type != device_type { continue; }
            }

            let context = Arc::new(
                Context::from_device(&cl_dev)
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?,
            );
            let queue = Arc::new(unsafe {
                CommandQueue::create(&context, cl_dev.id(), 0)
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
            });

            let dev = OpenCLDevice::new(cl_dev, context, queue)?;
            devices.push(Arc::new(dev));
        }
    }
    Ok(devices)
}
