use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use opencl3::program::Program;

use crate::cache::{ProgramCache, SharedProgramCache};
use crate::device::Device;
use crate::error::{CleError, Result};

const CUDA_DISABLED_ERROR: &str = "Error: CUDA is not enabled";
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: i32 = 12;
const CU_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 10;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: i32 = 9;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: i32 = 13;
const CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: i32 = 14;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: i32 = 2;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: i32 = 3;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: i32 = 4;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: i32 = 5;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: i32 = 6;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: i32 = 7;

/// Disabled CUDA device placeholder for CLIc API parity.
///
/// The Rust crate does not currently compile CUDA support, so lifecycle methods
/// report CUDA as unavailable while metadata accessors retain CLIc's CUDA names.
pub struct CUDADevice {
    cuda_device_index: usize,
    initialized: AtomicBool,
    wait_finish: AtomicBool,
    program_cache: SharedProgramCache,
}

impl CUDADevice {
    /// Canonical constructor alias for CLIc's `CUDADevice::CUDADevice()`.
    pub fn new(cuda_device_index: usize) -> Self {
        Self {
            cuda_device_index,
            initialized: AtomicBool::new(false),
            wait_finish: AtomicBool::new(true),
            program_cache: Arc::new(Mutex::new(ProgramCache::new())),
        }
    }

    /// Canonical alias for CLIc's `CUDADevice::initialize()`.
    pub fn initialize(&self) -> Result<()> {
        if self.is_initialized() {
            return Ok(());
        }
        let cuda_device_index = self.cuda_device_index;
        let device_available = false;
        if !device_available {
            return Err(CleError::Other(format!(
                "Error: Failed to get CUDA device at index {cuda_device_index}"
            )));
        }
        let context_created = false;
        if !context_created {
            return Err(CleError::Other(format!(
                "Error: Failed to create CUDA context for device {cuda_device_index}"
            )));
        }
        let stream_created = false;
        if !stream_created {
            self.initialized.store(false, Ordering::Relaxed);
            return Err(CleError::Other(format!(
                "Error: Failed to create CUDA stream for device {cuda_device_index}"
            )));
        }
        self.initialized.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Canonical alias for CLIc's `CUDADevice::finalize()`.
    pub fn finalize(&self) {
        if !self.is_initialized() {
            return;
        }
        self.wait_finish.store(true, Ordering::Relaxed);
        self.finish();
        self.initialized.store(false, Ordering::Relaxed);
    }

    /// Canonical alias for CLIc's `CUDADevice::setWaitToFinish()`.
    pub fn set_wait_to_finish(&self, flag: bool) {
        self.wait_finish.store(flag, Ordering::Relaxed);
    }

    /// Canonical alias for CLIc's `CUDADevice::isInitialized()`.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Relaxed)
    }

    /// Canonical alias for CLIc's `CUDADevice::getType()`.
    pub fn get_type(&self) -> &'static str {
        "CUDA"
    }

    /// Canonical alias for CLIc's `CUDADevice::getArch()`.
    pub fn get_arch(&self) -> String {
        let major = query_device_attribute(
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            self.cuda_device_index,
        );
        let minor = query_device_attribute(
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            self.cuda_device_index,
        );
        format!("{major}{minor}")
    }

    /// Canonical alias for CLIc's `CUDADevice::getNbDevicesFromContext()`.
    pub fn get_nb_devices_from_context(&self) -> usize {
        1
    }

    /// Canonical alias for CLIc's `CUDADevice::getDeviceIndex()`.
    pub fn get_device_index(&self) -> usize {
        self.cuda_device_index
    }

    /// Canonical alias for CLIc's `CUDADevice::getCUDADeviceIndex()`.
    pub fn get_cuda_device_index(&self) -> usize {
        self.cuda_device_index
    }

    /// Canonical alias for CLIc's `CUDADevice::getCUDADevice()`.
    pub fn get_cuda_device(&self) -> Result<()> {
        Err(CleError::Other(CUDA_DISABLED_ERROR.to_string()))
    }

    /// Canonical alias for CLIc's `CUDADevice::getCUDAContext()`.
    pub fn get_cuda_context(&self) -> Result<()> {
        Err(CleError::Other(CUDA_DISABLED_ERROR.to_string()))
    }

    /// Canonical alias for CLIc's `CUDADevice::getCUDAStream()`.
    pub fn get_cuda_stream(&self) -> Result<()> {
        Err(CleError::Other(CUDA_DISABLED_ERROR.to_string()))
    }

    /// Canonical alias for CLIc's `CUDADevice::getInfo()`.
    pub fn get_info(&self) -> String {
        let driver_version = 0;
        let total_global_mem = 0usize;

        let shared_mem_per_block = query_device_attribute(
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            self.cuda_device_index,
        );
        let regs_per_block = query_device_attribute(
            CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
            self.cuda_device_index,
        );
        let warp_size =
            query_device_attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE, self.cuda_device_index);
        let max_threads_per_block = query_device_attribute(
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            self.cuda_device_index,
        );
        let total_const_mem = query_device_attribute(
            CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
            self.cuda_device_index,
        );
        let major = query_device_attribute(
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            self.cuda_device_index,
        );
        let minor = query_device_attribute(
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            self.cuda_device_index,
        );
        let clock_rate =
            query_device_attribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, self.cuda_device_index);
        let texture_alignment = query_device_attribute(
            CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
            self.cuda_device_index,
        );
        let multi_proc_count = query_device_attribute(
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            self.cuda_device_index,
        );
        let max_block_dim_x =
            query_device_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, self.cuda_device_index);
        let max_block_dim_y =
            query_device_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, self.cuda_device_index);
        let max_block_dim_z =
            query_device_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, self.cuda_device_index);
        let max_grid_dim_x =
            query_device_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, self.cuda_device_index);
        let max_grid_dim_y =
            query_device_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, self.cuda_device_index);
        let max_grid_dim_z =
            query_device_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, self.cuda_device_index);

        let driver_version_f = driver_version as f32 / 1000.0;
        let mut result = format!(
            "({}) {} (driver {:.1})\n",
            self.get_type(),
            self.get_name(),
            driver_version_f
        );
        result.push_str(&info_line("Vendor:", "NVIDIA Corporation"));
        result.push_str(&info_line(
            "Device index:",
            &self.cuda_device_index.to_string(),
        ));
        result.push_str(&info_line(
            "Driver version:",
            &format!("{driver_version_f:.6}"),
        ));
        result.push_str(&info_line("Device type:", "GPU"));
        result.push_str(&info_line(
            "Multiprocessor count:",
            &multi_proc_count.to_string(),
        ));
        result.push_str(&info_line(
            "Total global memory:",
            &format!("{} MB", total_global_mem / (1024 * 1024)),
        ));
        result.push_str(&info_line(
            "Shared memory per block:",
            &format!("{} KB", shared_mem_per_block / 1024),
        ));
        result.push_str(&info_line(
            "Clock rate:",
            &format!("{} MHz", clock_rate / 1000),
        ));
        result.push_str(&info_line(
            "Total constant memory:",
            &format!("{} KB", total_const_mem / 1024),
        ));
        result.push_str(&info_line(
            "Registers per block:",
            &regs_per_block.to_string(),
        ));
        result.push_str(&info_line("Warp size:", &warp_size.to_string()));
        result.push_str(&info_line(
            "Max threads per block:",
            &max_threads_per_block.to_string(),
        ));
        result.push_str(&info_line(
            "Max block dimension:",
            &format!("{max_block_dim_x}, {max_block_dim_y}, {max_block_dim_z}"),
        ));
        result.push_str(&info_line(
            "Max grid dimension:",
            &format!("{max_grid_dim_x}, {max_grid_dim_y}, {max_grid_dim_z}"),
        ));
        result.push_str(&info_line(
            "Compute capability:",
            &format!("{major}.{minor}"),
        ));
        result.push_str(&info_line(
            "Texture alignment:",
            &texture_alignment.to_string(),
        ));
        result
    }

    /// Canonical alias for CLIc's `CUDADevice::getInfoExtended()`.
    pub fn get_info_extended(&self) -> String {
        self.get_info()
    }
}

impl Device for CUDADevice {
    fn get_name(&self) -> &str {
        "Unknown CUDA Device"
    }

    fn get_device_type(&self) -> &str {
        "gpu"
    }

    fn support_image(&self) -> bool {
        true
    }

    fn get_maximum_buffer_size(&self) -> usize {
        let total_mem = 0usize;
        let err = self.cuda_device_index == usize::MAX;
        if err {
            return 0;
        }
        total_mem
    }

    fn get_maximum_work_group_size(&self) -> usize {
        query_device_attribute(
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            self.cuda_device_index,
        ) as usize
    }

    fn get_local_memory_size(&self) -> usize {
        query_device_attribute(
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            self.cuda_device_index,
        ) as usize
    }

    fn finish(&self) {
        if !self.is_initialized() {
            return;
        }
        if self.wait_finish.load(Ordering::Relaxed) {}
    }

    fn get_program_from_cache(&self, key: &str) -> Option<Arc<Program>> {
        self.program_cache.lock().unwrap().get(key)
    }

    fn add_program_to_cache(&self, key: String, program: Arc<Program>) {
        self.program_cache.lock().unwrap().put(key, program);
    }

    fn device_hash(&self) -> String {
        crate::cache::DiskCache::hash(&self.get_info())
    }

    fn get_platform(&self) -> String {
        "NVIDIA".to_string()
    }
}

/// Disabled equivalent of CLIc's CUDA `queryDeviceAttribute()` helper.
fn query_device_attribute(_attrib: i32, _device: usize) -> i32 {
    let value = 0;
    let err = _device == usize::MAX;
    if err {
        return value;
    }
    value
}

/// Format a labelled CUDA info line like CLIc's `infoLine()` helper.
fn info_line(label: &str, value: &str) -> String {
    format!("\t{label:<29}{value}\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_device_reports_disabled_lifecycle() {
        let device = CUDADevice::new(3);
        assert!(!device.is_initialized());
        assert!(device.initialize().is_err());
        assert_eq!(device.get_type(), "CUDA");
        assert_eq!(device.get_cuda_device_index(), 3);
        assert_eq!(device.get_platform(), "NVIDIA");
    }

    #[test]
    fn cuda_info_line_matches_label_padding() {
        assert_eq!(
            info_line("Vendor:", "NVIDIA"),
            "\tVendor:                      NVIDIA\n"
        );
    }
}
