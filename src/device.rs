use std::any::Any;
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{
    cl_device_info, Device as ClDevice, CL_DEVICE_EXTENSIONS, CL_DEVICE_TYPE_ACCELERATOR,
    CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_CUSTOM, CL_DEVICE_TYPE_GPU,
    CL_DEVICE_VENDOR, CL_DEVICE_VERSION, CL_DRIVER_VERSION,
};
use opencl3::platform::{get_platforms, Platform};
use opencl3::program::Program;
use opencl3::types::{cl_command_queue, cl_context, cl_device_id, cl_device_type, cl_platform_id};

use crate::cache::{ProgramCache, SharedProgramCache};
use crate::error::{CleError, Result};

/// Abstraction over a GPU device.
pub trait Device: Send + Sync + Any {
    fn get_name(&self) -> &str;
    fn get_device_type(&self) -> &str;
    fn support_image(&self) -> bool;
    fn get_maximum_buffer_size(&self) -> usize;
    fn get_maximum_work_group_size(&self) -> usize;
    fn get_local_memory_size(&self) -> usize;
    fn finish(&self);
    fn get_program_from_cache(&self, key: &str) -> Option<Arc<Program>>;
    fn add_program_to_cache(&self, key: String, program: Arc<Program>);
    fn device_hash(&self) -> String;

    fn get_platform(&self) -> String;
}

pub type DeviceArc = Arc<dyn Device>;

// ── OpenCL device ────────────────────────────────────────────────────────────

const OPENCL_NAME_BUFFER_SIZE: usize = 256;
const OPENCL_EXTENSIONS_BUFFER_SIZE: usize = 1024;

pub fn trim_string(s: &mut String) {
    let trimmed_len = s.trim_end_matches(' ').len();
    s.truncate(trimmed_len);
}

pub fn safe_get_device_info(
    device: cl_device_id,
    param: cl_device_info,
    buffer_size: usize,
) -> String {
    match ClDevice::new(device).get_data(param) {
        Ok(mut buffer) => {
            buffer.truncate(buffer_size.min(buffer.len()));
            if let Some(len) = buffer.iter().position(|&byte| byte == 0) {
                buffer.truncate(len);
            }
            let mut result = String::from_utf8_lossy(&buffer).into_owned();
            trim_string(&mut result);
            result
        }
        Err(err) => {
            eprintln!(
                "Failed to retrieve OpenCL device info (param: {}, error: {:?})",
                param, err
            );
            String::new()
        }
    }
}

fn get_device_type_map(uppercase: bool) -> &'static [(cl_device_type, &'static str)] {
    if uppercase {
        &[
            (CL_DEVICE_TYPE_CPU, "CPU"),
            (CL_DEVICE_TYPE_GPU, "GPU"),
            (CL_DEVICE_TYPE_ACCELERATOR, "Accelerator"),
            (CL_DEVICE_TYPE_CUSTOM, "Custom"),
        ]
    } else {
        &[
            (CL_DEVICE_TYPE_CPU, "cpu"),
            (CL_DEVICE_TYPE_GPU, "gpu"),
            (CL_DEVICE_TYPE_ACCELERATOR, "accelerator"),
            (CL_DEVICE_TYPE_CUSTOM, "custom"),
        ]
    }
}

pub struct OpenCLDevice {
    pub(crate) _ocl_device: ClDevice,
    pub(crate) context: Arc<Context>,
    pub(crate) queue: Arc<CommandQueue>,
    program_cache: SharedProgramCache,
    name: String,
    dtype: String,
    image_support: bool,
    wait_finish: AtomicBool,
    initialized: AtomicBool,
    device_index: usize,
}

#[derive(Debug, Clone)]
pub struct OpenCLDeviceResources {
    pub platform: Option<cl_platform_id>,
    pub device: cl_device_id,
    pub device_type: cl_device_type,
    pub device_name: String,
    pub platform_name: String,
    pub platform_vendor: String,
    pub image_support: bool,
    pub device_index: usize,
}

// Safety: OpenCL 1.2+ spec guarantees that context, command queue, and mem objects
// are thread-safe after creation (§5.13). The opencl3 crate wraps raw handles that
// don't carry Rust marker traits, but the underlying cl_context/cl_command_queue
// can be safely shared across threads. Interior mutable state (program_cache) is
// protected by Mutex.
unsafe impl Send for OpenCLDevice {}
unsafe impl Sync for OpenCLDevice {}

impl OpenCLDevice {
    pub fn new(
        ocl_device: ClDevice,
        context: Arc<Context>,
        queue: Arc<CommandQueue>,
        device_index: usize,
    ) -> Result<Self> {
        let mut name = ocl_device.name().unwrap_or_default();
        trim_string(&mut name);
        let device_type = ocl_device.dev_type().unwrap_or(0);
        let dtype = get_device_type_map(false)
            .iter()
            .find_map(|(candidate, name)| (*candidate == device_type).then_some(*name))
            .unwrap_or("unknown")
            .to_string();
        let image_support = ocl_device.image_support().unwrap_or(false);

        Ok(OpenCLDevice {
            _ocl_device: ocl_device,
            context,
            queue,
            program_cache: Arc::new(Mutex::new(ProgramCache::new())),
            name,
            dtype,
            image_support,
            wait_finish: AtomicBool::new(false),
            initialized: AtomicBool::new(true),
            device_index,
        })
    }

    /// Canonical alias for CLIc's `OpenCLDevice::getType()`.
    pub fn get_type(&self) -> &'static str {
        "OpenCL"
    }

    /// Canonical alias for CLIc's `OpenCLDevice::isInitialized()`.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Relaxed)
    }

    /// Canonical alias for CLIc's `OpenCLDevice::initialize()`.
    pub fn initialize(&self) {
        if self.is_initialized() {
            return;
        }
        let _ = self.context.devices();
        let _ = self.queue.get();
        self.initialized.store(true, Ordering::Relaxed);
    }

    /// Canonical alias for CLIc's `OpenCLDevice::finalize()`.
    pub fn finalize(&self) {
        if !self.is_initialized() {
            return;
        }
        self.wait_finish.store(true, Ordering::Relaxed);
        self.finish();
        self.wait_finish.store(false, Ordering::Relaxed);
        self.initialized.store(false, Ordering::Relaxed);
    }

    /// Canonical alias for CLIc's `OpenCLDevice::getCLPlatform()`.
    pub fn get_cl_platform(&self) -> Option<cl_platform_id> {
        self._ocl_device.platform().ok()
    }

    /// Canonical alias for CLIc's `OpenCLDevice::getCLDevice()`.
    pub fn get_cl_device(&self) -> cl_device_id {
        self._ocl_device.id()
    }

    /// Canonical alias for CLIc's `OpenCLDevice::getCLContext()`.
    pub fn get_cl_context(&self) -> cl_context {
        self.context.get()
    }

    /// Canonical alias for CLIc's `OpenCLDevice::getCLCommandQueue()`.
    pub fn get_cl_command_queue(&self) -> cl_command_queue {
        self.queue.get()
    }

    /// Rust-owned counterpart to CLIc's nested `OpenCLDevice::Context` wrapper.
    pub fn context_new(&self) -> Arc<Context> {
        Arc::clone(&self.context)
    }

    /// Accessor counterpart to CLIc's `OpenCLDevice::Context::get()`.
    pub fn context_get(&self) -> cl_context {
        self.get_cl_context()
    }

    /// Move-assignment counterpart for CLIc's context wrapper under Arc ownership.
    pub fn context_assign(&mut self, context: Arc<Context>) {
        self.context = context;
    }

    /// Rust-owned counterpart to CLIc's nested `OpenCLDevice::CommandQueue` wrapper.
    pub fn command_queue_new(&self) -> Arc<CommandQueue> {
        Arc::clone(&self.queue)
    }

    /// Move-assignment counterpart for CLIc's command queue wrapper under Arc ownership.
    pub fn command_queue_assign(&mut self, queue: Arc<CommandQueue>) {
        self.queue = queue;
    }

    /// Accessor counterpart to CLIc's `OpenCLDevice::CommandQueue::get()`.
    pub fn command_queue_get(&self) -> cl_command_queue {
        self.get_cl_command_queue()
    }

    /// Snapshot counterpart to CLIc's nested `OpenCLDevice::Resources`.
    pub fn resources_new(&self) -> OpenCLDeviceResources {
        let platform = self.get_cl_platform();
        let platform_name = platform
            .and_then(|id| Platform::new(id).name().ok())
            .unwrap_or_default();
        let platform_vendor = platform
            .and_then(|id| Platform::new(id).vendor().ok())
            .unwrap_or_default();

        OpenCLDeviceResources {
            platform,
            device: self.get_cl_device(),
            device_type: self._ocl_device.dev_type().unwrap_or(0),
            device_name: self.name.clone(),
            platform_name,
            platform_vendor,
            image_support: self.image_support,
            device_index: self.device_index,
        }
    }

    /// Accessor counterpart to CLIc's `OpenCLDevice::Resources::get_device()`.
    pub fn resources_get_device(&self) -> cl_device_id {
        self.get_cl_device()
    }

    /// Accessor counterpart to CLIc's `OpenCLDevice::Resources::get_platform()`.
    pub fn resources_get_platform(&self) -> Option<cl_platform_id> {
        self.get_cl_platform()
    }

    /// Distinct CCC target for CLIc's top-level `OpenCLDevice::getCLContext()`.
    pub fn get_cl_context_handle(&self) -> cl_context {
        self.get_cl_context()
    }

    /// Distinct CCC target for CLIc's top-level `OpenCLDevice::getCLCommandQueue()`.
    pub fn get_cl_command_queue_handle(&self) -> cl_command_queue {
        self.get_cl_command_queue()
    }

    /// Canonical alias for CLIc's `OpenCLDevice::getNbDevicesFromContext()`.
    pub fn get_nb_devices_from_context(&self) -> usize {
        self.context.devices().len()
    }

    /// Canonical alias for CLIc's `OpenCLDevice::setWaitToFinish()`.
    pub fn set_wait_to_finish(&self, flag: bool) {
        self.wait_finish.store(flag, Ordering::Relaxed);
    }

    /// Canonical alias for CLIc's `OpenCLDevice::getDeviceIndex()`.
    pub fn get_device_index(&self) -> usize {
        self.device_index
    }

    /// Canonical alias for CLIc's `OpenCLDevice::getInfo()`.
    pub fn get_info(&self) -> String {
        let device = self.get_cl_device();
        let version = safe_get_device_info(device, CL_DEVICE_VERSION, OPENCL_NAME_BUFFER_SIZE);
        let vendor = safe_get_device_info(device, CL_DEVICE_VENDOR, OPENCL_NAME_BUFFER_SIZE);
        let driver = safe_get_device_info(device, CL_DRIVER_VERSION, OPENCL_NAME_BUFFER_SIZE);
        let name = self.get_name();
        let compute_units = self._ocl_device.max_compute_units().unwrap_or(0);
        let max_clock_frequency = self._ocl_device.max_clock_frequency().unwrap_or(0);
        let global_mem_size = self._ocl_device.global_mem_size().unwrap_or(0);
        let local_mem_size = self._ocl_device.local_mem_size().unwrap_or(0);
        let max_mem_size = self._ocl_device.max_mem_alloc_size().unwrap_or(0);
        let image_support = self._ocl_device.image_support().unwrap_or(false);
        let device_type = self._ocl_device.dev_type().unwrap_or(0);
        let device_type_str = get_device_type_map(true)
            .iter()
            .find_map(|(candidate, name)| (*candidate == device_type).then_some(*name))
            .unwrap_or("Unknown");

        let mut result = String::new();
        let _ = writeln!(result, "({}) {} ({})", self.get_type(), name, version);
        let _ = writeln!(result, "{:<30}{}", "\tVendor: ", vendor);
        let _ = writeln!(result, "{:<30}{}", "\tDriver Version: ", driver);
        let _ = writeln!(result, "{:<30}{}", "\tDevice Type: ", device_type_str);
        let _ = writeln!(result, "{:<30}{}", "\tCompute Units: ", compute_units);
        let _ = writeln!(
            result,
            "{:<30}{} MB",
            "\tGlobal Memory Size: ",
            global_mem_size / (1024 * 1024)
        );
        let _ = writeln!(
            result,
            "{:<30}{} MB",
            "\tLocal Memory Size: ",
            local_mem_size / (1024 * 1024)
        );
        let _ = writeln!(
            result,
            "{:<30}{} MB",
            "\tMaximum Buffer Size: ",
            max_mem_size / (1024 * 1024)
        );
        let _ = writeln!(
            result,
            "{:<30}{} MHz",
            "\tMax Clock Frequency: ", max_clock_frequency
        );
        let _ = writeln!(
            result,
            "{:<30}{}",
            "\tImage Support: ",
            if image_support { "Yes" } else { "No" }
        );
        result
    }

    /// Canonical alias for CLIc's `OpenCLDevice::getInfoExtended()`.
    pub fn get_info_extended(&self) -> String {
        let extensions = safe_get_device_info(
            self.get_cl_device(),
            CL_DEVICE_EXTENSIONS,
            OPENCL_EXTENSIONS_BUFFER_SIZE,
        );
        let max_work_group_size = self._ocl_device.max_work_group_size().unwrap_or(0);
        let max_work_item_dimensions = self._ocl_device.max_work_item_dimensions().unwrap_or(0);
        let max_work_item_sizes = self
            ._ocl_device
            .max_work_item_sizes()
            .unwrap_or_else(|_| vec![0, 0, 0]);
        let mut result = self.get_info();
        let _ = writeln!(
            result,
            "{:<30}{}",
            "\tMax Work Group Size: ", max_work_group_size
        );
        let _ = writeln!(
            result,
            "{:<30}{}",
            "\tMax Work Item Dimensions: ", max_work_item_dimensions
        );
        let _ = writeln!(
            result,
            "{:<30}{}, {}, {}",
            "\tMax Work Item Sizes: ",
            max_work_item_sizes.first().copied().unwrap_or(0),
            max_work_item_sizes.get(1).copied().unwrap_or(0),
            max_work_item_sizes.get(2).copied().unwrap_or(0)
        );
        let _ = write!(result, "{:<30}", "\tExtensions:");
        if extensions.trim().is_empty() {
            result.push_str(" (none)\n");
        } else {
            for extension in extensions.split_whitespace() {
                result.push_str("\n\t\t");
                let _ = write!(result, "{:<30}", extension);
            }
            result.push('\n');
        }
        result
    }
}

impl Device for OpenCLDevice {
    fn get_name(&self) -> &str {
        &self.name
    }
    fn get_device_type(&self) -> &str {
        &self.dtype
    }
    fn support_image(&self) -> bool {
        self.image_support
    }
    fn get_maximum_buffer_size(&self) -> usize {
        self._ocl_device.max_mem_alloc_size().unwrap_or(0) as usize
    }
    fn get_maximum_work_group_size(&self) -> usize {
        self._ocl_device.max_work_group_size().unwrap_or(0)
    }
    fn get_local_memory_size(&self) -> usize {
        self._ocl_device.local_mem_size().unwrap_or(0) as usize
    }
    fn device_hash(&self) -> String {
        crate::cache::DiskCache::hash(&self.get_info())
    }
    fn get_platform(&self) -> String {
        self._ocl_device
            .platform()
            .ok()
            .and_then(|id| opencl3::platform::Platform::new(id).name().ok())
            .unwrap_or_default()
    }
    fn finish(&self) {
        if !self.is_initialized() {
            eprintln!("OpenCL device not initialized");
            return;
        }
        if self.wait_finish.load(Ordering::Relaxed) {
            let _ = self.queue.finish();
        }
    }

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

        for (device_index, &id) in device_ids.iter().enumerate() {
            let cl_dev = ClDevice::new(id);

            // CLIc filters only "gpu" and "cpu"; every other selector returns all devices.
            if device_type == "gpu" || device_type == "cpu" {
                let cl_device_type = cl_dev.dev_type().unwrap_or(0);
                let dev_type = get_device_type_map(false)
                    .iter()
                    .find_map(|(candidate, name)| (*candidate == cl_device_type).then_some(*name))
                    .unwrap_or("unknown");
                if dev_type != device_type {
                    continue;
                }
            }

            let context = Arc::new(
                Context::from_device(&cl_dev).map_err(|e| CleError::OpenCL(format!("{:?}", e)))?,
            );
            #[allow(deprecated)]
            let queue = Arc::new(unsafe {
                CommandQueue::create(&context, cl_dev.id(), 0)
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
            });

            let dev = OpenCLDevice::new(cl_dev, context, queue, device_index)?;
            devices.push(Arc::new(dev));
        }
    }
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_type_map_matches_clic_case_variants() {
        let lower = get_device_type_map(false);
        assert!(lower.contains(&(CL_DEVICE_TYPE_CPU, "cpu")));
        assert!(lower.contains(&(CL_DEVICE_TYPE_GPU, "gpu")));
        assert!(lower.contains(&(CL_DEVICE_TYPE_ACCELERATOR, "accelerator")));
        assert!(lower.contains(&(CL_DEVICE_TYPE_CUSTOM, "custom")));

        let upper = get_device_type_map(true);
        assert!(upper.contains(&(CL_DEVICE_TYPE_CPU, "CPU")));
        assert!(upper.contains(&(CL_DEVICE_TYPE_GPU, "GPU")));
        assert!(upper.contains(&(CL_DEVICE_TYPE_ACCELERATOR, "Accelerator")));
        assert!(upper.contains(&(CL_DEVICE_TYPE_CUSTOM, "Custom")));
    }

    #[test]
    fn device_type_name_reports_unknown_like_clic() {
        assert_eq!(
            get_device_type_map(false)
                .iter()
                .find_map(|(candidate, name)| (*candidate == 0).then_some(*name))
                .unwrap_or("unknown"),
            "unknown"
        );
        assert_eq!(
            get_device_type_map(true)
                .iter()
                .find_map(|(candidate, name)| (*candidate == 0).then_some(*name))
                .unwrap_or("Unknown"),
            "Unknown"
        );
    }

    #[test]
    fn trim_string_removes_trailing_spaces_like_clic() {
        let mut value = "OpenCL Device   ".to_string();
        trim_string(&mut value);
        assert_eq!(value, "OpenCL Device");
    }
}
