use crate::backend::{Backend, GpuMemPtr, KernelArg};
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::types::{DType, MType};

const CUDA_DISABLED_ERROR: &str = "Error: CUDA is not enabled";

macro_rules! cuda_disabled {
    () => {
        Err(CleError::Other(CUDA_DISABLED_ERROR.to_string()))
    };
}

/// Disabled CUDA backend placeholder for CLIc API parity.
pub struct CUDABackend;

impl CUDABackend {
    /// Canonical constructor alias for CLIc's `CUDABackend::CUDABackend()`.
    pub fn new() -> Self {
        Self
    }

    /// Canonical alias for CLIc's `CUDABackend::getDevices()`.
    pub fn get_devices(&self, _device_type: &str) -> Result<Vec<DeviceArc>> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::getDevice()`.
    pub fn get_device(&self, _name: &str, _device_type: &str) -> Result<DeviceArc> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::getDeviceFromIndex()`.
    pub fn get_device_from_index(&self, _index: usize, _device_type: &str) -> Result<DeviceArc> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::getDevicesList()`.
    pub fn get_devices_list(&self, _device_type: &str) -> Result<Vec<String>> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::getType()`.
    pub fn get_type(&self) -> &'static str {
        "CUDA"
    }

    /// Canonical alias for CLIc's `CUDABackend::allocateBuffer()`.
    pub fn allocate_buffer(&self, _device: &DeviceArc, _byte_size: usize) -> Result<GpuMemPtr> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::allocateImage()`.
    pub fn allocate_image(
        &self,
        _device: &DeviceArc,
        _region: [usize; 3],
        _dtype: DType,
    ) -> Result<GpuMemPtr> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::freeMemory()`.
    pub fn free_memory(&self, _device: &DeviceArc, _mtype: MType, _mem: GpuMemPtr) -> Result<()> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::getRefCount()`.
    pub fn get_ref_count(&self, _mem: &GpuMemPtr) -> Result<usize> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::writeBuffer()`.
    pub fn write_buffer(
        &self,
        _device: &DeviceArc,
        _mem: &GpuMemPtr,
        _buffer_shape: [usize; 3],
        _buffer_origin: [usize; 3],
        _region: [usize; 3],
        _host_data: &[u8],
    ) -> Result<()> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::readBuffer()`.
    pub fn read_buffer(
        &self,
        _device: &DeviceArc,
        _mem: &GpuMemPtr,
        _buffer_shape: [usize; 3],
        _buffer_origin: [usize; 3],
        _region: [usize; 3],
        _host_data: &mut [u8],
    ) -> Result<()> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::copyMemoryImageToBuffer()`.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_memory_image_to_buffer(
        &self,
        _device: &DeviceArc,
        _src: &GpuMemPtr,
        _src_origin: [usize; 3],
        _src_shape: [usize; 3],
        _dst: &GpuMemPtr,
        _dst_origin: [usize; 3],
        _dst_shape: [usize; 3],
        _region: [usize; 3],
        _item_size: usize,
    ) -> Result<()> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::copyMemoryBufferToImage()`.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_memory_buffer_to_image(
        &self,
        _device: &DeviceArc,
        _src: &GpuMemPtr,
        _src_origin: [usize; 3],
        _src_shape: [usize; 3],
        _dst: &GpuMemPtr,
        _dst_origin: [usize; 3],
        _dst_shape: [usize; 3],
        _region: [usize; 3],
        _item_size: usize,
    ) -> Result<()> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::copyMemoryImageToImage()`.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_memory_image_to_image(
        &self,
        _device: &DeviceArc,
        _src: &GpuMemPtr,
        _src_origin: [usize; 3],
        _src_shape: [usize; 3],
        _dst: &GpuMemPtr,
        _dst_origin: [usize; 3],
        _dst_shape: [usize; 3],
        _region: [usize; 3],
        _item_size: usize,
    ) -> Result<()> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::setBuffer()`.
    pub fn set_buffer(
        &self,
        _device: &DeviceArc,
        _mem: &GpuMemPtr,
        _buffer_shape: [usize; 3],
        _buffer_origin: [usize; 3],
        _region: [usize; 3],
        _dtype: DType,
        _value: f32,
    ) -> Result<()> {
        cuda_disabled!()
    }

    /// Canonical alias for CLIc's `CUDABackend::buildKernel()`.
    pub fn build_kernel(
        &self,
        _device: &DeviceArc,
        _kernel_source: &str,
        _kernel_name: &str,
    ) -> Result<()> {
        cuda_disabled!()
    }
}

impl Drop for CUDABackend {
    fn drop(&mut self) {}
}

impl Backend for CUDABackend {
    fn allocate_memory(
        &self,
        _device: &DeviceArc,
        _region: [usize; 3],
        _dtype: DType,
        _mtype: MType,
    ) -> Result<GpuMemPtr> {
        cuda_disabled!()
    }

    fn write_memory(&self, _device: &DeviceArc, _mem: &GpuMemPtr, _data: &[u8]) -> Result<()> {
        cuda_disabled!()
    }

    fn read_memory(&self, _device: &DeviceArc, _mem: &GpuMemPtr, _data: &mut [u8]) -> Result<()> {
        cuda_disabled!()
    }

    fn copy_memory_buffer_to_buffer(
        &self,
        _device: &DeviceArc,
        _src: &GpuMemPtr,
        _dst: &GpuMemPtr,
        _byte_size: usize,
    ) -> Result<()> {
        cuda_disabled!()
    }

    fn copy_memory_buffer_to_buffer_region(
        &self,
        _device: &DeviceArc,
        _src: &GpuMemPtr,
        _src_origin: [usize; 3],
        _src_shape: [usize; 3],
        _dst: &GpuMemPtr,
        _dst_origin: [usize; 3],
        _dst_shape: [usize; 3],
        _region: [usize; 3],
        _item_size: usize,
    ) -> Result<()> {
        cuda_disabled!()
    }

    fn set_memory(
        &self,
        _device: &DeviceArc,
        _mem: &GpuMemPtr,
        _value: f32,
        _dtype: DType,
        _element_count: usize,
    ) -> Result<()> {
        cuda_disabled!()
    }

    fn execute_kernel(
        &self,
        _device: &DeviceArc,
        _source: &str,
        _kernel_name: &str,
        _global_range: [usize; 3],
        _local_range: [usize; 3],
        _args: &[KernelArg],
    ) -> Result<()> {
        cuda_disabled!()
    }

    fn get_preamble(&self) -> &'static str {
        ""
    }
}

/// Disabled equivalent of CLIc's `CUDAContextGuard`.
pub struct CUDAContextGuard {
    device: DeviceArc,
}

impl CUDAContextGuard {
    pub fn new(device: DeviceArc) -> Result<Self> {
        let _ = device;
        cuda_disabled!()
    }

    pub fn device(&self) -> &DeviceArc {
        &self.device
    }

    pub fn context(&self) -> Result<()> {
        cuda_disabled!()
    }

    pub fn stream(&self) -> Result<()> {
        cuda_disabled!()
    }
}

/// Cache PTX bytes using the same disk-cache layout as OpenCL binaries.
pub fn save_ptx_to_cache(device_hash: &str, source_hash: &str, ptx: &str) {
    crate::cache::DiskCache::instance().save_binary(
        device_hash,
        source_hash,
        "ptx",
        ptx.as_bytes(),
    );
}

/// Load PTX bytes from the disk cache.
pub fn load_ptx_from_cache(device_hash: &str, source_hash: &str) -> String {
    crate::cache::DiskCache::instance()
        .load_binary(device_hash, source_hash, "ptx")
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

/// Safe bit-preserving cast helper for equal-sized integer/float values.
pub fn bit_cast(value: u32) -> f32 {
    f32::from_bits(value)
}

/// Disabled equivalent of CLIc's CUDA `performMemcpy()` helper.
pub fn perform_memcpy() -> Result<()> {
    let shape = [1usize, 1, 1];
    let src_origin = [0usize, 0, 0];
    let dst_origin = [0usize, 0, 0];
    let region = [1usize, 1, 1];
    let need_3d = shape[2] > 1 || src_origin[2] > 0 || dst_origin[2] > 0;
    let need_2d = !need_3d && (shape[1] > 1 || src_origin[1] > 0 || dst_origin[1] > 0);
    let copy_kind = if need_3d {
        "3D"
    } else if need_2d {
        "2D"
    } else {
        "1D"
    };
    let _width_in_bytes = region[0];
    let _height = region[1];
    let _depth = region[2];
    Err(CleError::Other(format!(
        "{CUDA_DISABLED_ERROR}: cannot perform CUDA {copy_kind} memory copy"
    )))
}

/// Disabled equivalent of CLIc's NVRTC `compileToPtx()` helper.
pub fn compile_to_ptx(kernel_source: &str, arch: &str) -> Result<String> {
    let arch_opt = format!("--gpu-architecture=compute_{arch}");
    let warn_opt = "--disable-warnings";
    let options = [arch_opt.as_str(), warn_opt];
    if kernel_source.trim().is_empty() {
        return Err(CleError::Other(
            "Error: Failed to create NVRTC program from empty source".to_string(),
        ));
    }
    let compile_result = Err::<(), _>(CleError::Other(format!(
        "{CUDA_DISABLED_ERROR}: NVRTC unavailable for options {} {}",
        options[0], options[1]
    )));
    if let Err(err) = compile_result {
        let build_log = "NVRTC is not compiled into this build";
        return Err(CleError::Other(format!(
            "Error: Failed to compile kernel. {err}\nBuild log:\n{build_log}"
        )));
    }
    cuda_disabled!()
}

/// Compute CUDA block dimensions using CLIc's current heuristic.
pub fn compute_block_size(global_size: [usize; 3]) -> [usize; 3] {
    let dim = global_size.iter().filter(|&&size| size > 1).count();
    let mut block = [1usize, 1, 1];

    match dim {
        0 => {}
        1 => {
            for (axis, size) in global_size.iter().copied().enumerate() {
                if size > 1 {
                    block[axis] = size.min(256);
                }
            }
        }
        2 => {
            let mut first_active = true;
            for (axis, size) in global_size.iter().copied().enumerate() {
                if size > 1 {
                    block[axis] = if first_active {
                        first_active = false;
                        size.min(32)
                    } else {
                        size.min(8)
                    };
                }
            }
        }
        _ => {
            let preferred = [32usize, 8, 1];
            for axis in 0..3 {
                block[axis] = global_size[axis].min(preferred[axis]);
            }
        }
    }

    block[2] = block[2].min(64);
    while block[0] * block[1] * block[2] > 1024 {
        let axis = (0..3).max_by_key(|&axis| block[axis]).unwrap();
        block[axis] = block[axis].div_ceil(2);
    }
    block
}

/// Compute CUDA grid dimensions by ceiling-dividing global size by block size.
pub fn compute_grid_size(global_size: [usize; 3], block_size: [usize; 3]) -> [usize; 3] {
    [
        global_size[0].div_ceil(block_size[0]),
        global_size[1].div_ceil(block_size[1]),
        global_size[2].div_ceil(block_size[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_backend_reports_disabled_operations() {
        let backend = CUDABackend::new();
        assert_eq!(backend.get_type(), "CUDA");
        assert!(backend.get_devices("gpu").is_err());
        assert_eq!(backend.get_preamble(), "");
    }

    #[test]
    fn cuda_launch_geometry_matches_clic_heuristic() {
        assert_eq!(compute_block_size([128, 1, 1]), [128, 1, 1]);
        assert_eq!(compute_block_size([256, 128, 1]), [32, 8, 1]);
        assert_eq!(compute_block_size([256, 128, 8]), [32, 8, 1]);
        assert_eq!(compute_grid_size([257, 17, 1], [32, 8, 1]), [9, 3, 1]);
    }
}
