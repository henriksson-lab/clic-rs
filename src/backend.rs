use std::any::Any;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};

use opencl3::command_queue::CommandQueue;
use opencl3::kernel::Kernel;
use opencl3::memory::{
    Buffer, ClMem, Image, CL_FLOAT, CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE2D,
    CL_MEM_OBJECT_IMAGE3D, CL_MEM_READ_WRITE, CL_R, CL_SIGNED_INT16, CL_SIGNED_INT32,
    CL_SIGNED_INT8, CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_UNSIGNED_INT8,
};
use opencl3::program::Program;
use opencl3::types::{cl_channel_type, cl_image_desc, cl_image_format, cl_mem, CL_TRUE};

use crate::device::{DeviceArc, OpenCLDevice};
use crate::error::{CleError, Result};
use crate::types::{to_bytes, DType, MType};

// ── GPU memory handle ────────────────────────────────────────────────────────

/// Erased GPU memory — wraps `Mutex<Buffer<u8>>` for buffers.
/// Mutex provides interior mutability needed by `enqueue_write_buffer(&mut Buffer<T>)`.
pub enum GpuMemory {
    Buffer(Mutex<Buffer<u8>>),
    Image(Mutex<Image>),
}

pub type GpuMemPtr = Arc<GpuMemory>;

// Safety: The underlying cl_mem handle is thread-safe per OpenCL 1.2+ spec (§5.13).
// The Mutex<Buffer<u8>> wrapper provides synchronized access for enqueue operations.
unsafe impl Send for GpuMemory {}
unsafe impl Sync for GpuMemory {}

pub fn to_image_channel_type(dtype: DType) -> cl_channel_type {
    match dtype {
        DType::Float => CL_FLOAT,
        DType::Int32 => CL_SIGNED_INT32,
        DType::Uint32 => CL_UNSIGNED_INT32,
        DType::Int16 => CL_SIGNED_INT16,
        DType::Uint16 => CL_UNSIGNED_INT16,
        DType::Int8 => CL_SIGNED_INT8,
        DType::Uint8 => CL_UNSIGNED_INT8,
        DType::Complex | DType::Unknown => CL_FLOAT,
    }
}

pub struct BufferPitches {
    pub row: usize,
    pub slice: usize,
}

pub fn compute_buffer_pitches(shape: [usize; 3]) -> BufferPitches {
    BufferPitches {
        row: if shape[1] > 1 { shape[0] } else { 0 },
        slice: if shape[2] > 1 { shape[0] * shape[1] } else { 0 },
    }
}

pub fn fill_buffer_typed<T: Copy>(
    queue: &CommandQueue,
    buffer: &mut Buffer<u8>,
    size_bytes: usize,
    value: T,
) -> Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts((&value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    };
    unsafe {
        queue
            .enqueue_fill_buffer(buffer, bytes, 0, size_bytes, &[])
            .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
    };
    Ok(())
}

pub fn fill_image_typed<T: Copy>(
    queue: &CommandQueue,
    image: &mut Image,
    origin: [usize; 3],
    region: [usize; 3],
    value: T,
) -> Result<()> {
    let fill_color = [value; 4];
    unsafe {
        queue
            .enqueue_fill_image(
                image,
                fill_color.as_ptr().cast::<c_void>(),
                origin.as_ptr(),
                region.as_ptr(),
                &[],
            )
            .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
    };
    Ok(())
}

pub fn wrap_mem_object(buffer: Buffer<u8>) -> GpuMemPtr {
    Arc::new(GpuMemory::Buffer(Mutex::new(buffer)))
}

pub fn wrap_program(program: Program) -> Arc<Program> {
    Arc::new(program)
}

pub fn wrap_kernel(kernel: Kernel) -> Kernel {
    kernel
}

pub fn build_program(opencl_device: &OpenCLDevice, program: &mut Program) -> Result<()> {
    program
        .build(&[opencl_device.get_cl_device()], "-w")
        .map_err(|e| {
            let log = program
                .get_build_log(opencl_device.get_cl_device())
                .unwrap_or_default();
            if log.is_empty() {
                CleError::OpenCL(format!("Error: Failed to build OpenCL program: {:?}", e))
            } else {
                CleError::OpenCL(format!(
                    "Error: Failed to build OpenCL program: {:?}\nBuild log:\n{}",
                    e, log
                ))
            }
        })
}

pub fn save_binary_to_cache(
    device_hash: &str,
    source_hash: &str,
    program: &Program,
    _device: &DeviceArc,
) {
    if let Ok(binaries) = program.get_binaries() {
        if let Some(bin) = binaries.first() {
            crate::cache::DiskCache::instance().save_binary(device_hash, source_hash, "bin", bin);
        }
    }
}

pub fn load_program_from_cache(
    opencl_device: &OpenCLDevice,
    device_hash: &str,
    source_hash: &str,
) -> Option<Arc<Program>> {
    let binary =
        crate::cache::DiskCache::instance().load_binary(device_hash, source_hash, "bin")?;
    let mut program = unsafe {
        Program::create_from_binary(
            &opencl_device.context,
            &[opencl_device.get_cl_device()],
            &[binary.as_slice()],
        )
        .ok()?
    };
    if let Err(e) = build_program(opencl_device, &mut program) {
        eprintln!("Warning: Failed to build program from cached binary: {}", e);
        return None;
    }
    Some(wrap_program(program))
}

pub fn create_program_from_source(
    opencl_device: &OpenCLDevice,
    kernel_source: &str,
) -> Result<Arc<Program>> {
    let mut program =
        Program::create_from_source(&opencl_device.context, kernel_source).map_err(|e| {
            CleError::OpenCL(format!(
                "Error: Failed to create program from source: {:?}",
                e
            ))
        })?;
    build_program(opencl_device, &mut program)?;
    Ok(wrap_program(program))
}

// ── Backend trait ────────────────────────────────────────────────────────────

pub trait Backend: Send + Sync {
    fn allocate_memory(
        &self,
        device: &DeviceArc,
        region: [usize; 3],
        dtype: DType,
        mtype: MType,
    ) -> Result<GpuMemPtr>;

    fn write_memory(&self, device: &DeviceArc, mem: &GpuMemPtr, data: &[u8]) -> Result<()>;

    fn read_memory(&self, device: &DeviceArc, mem: &GpuMemPtr, data: &mut [u8]) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn write_memory_region(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        buffer_shape: [usize; 3],
        buffer_origin: [usize; 3],
        region: [usize; 3],
        dtype: DType,
        mtype: MType,
        data: &[u8],
    ) -> Result<()> {
        let _ = (
            device,
            mem,
            buffer_shape,
            buffer_origin,
            region,
            dtype,
            mtype,
            data,
        );
        Err(CleError::Other(
            "memory region write is not implemented for this backend".to_string(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn read_memory_region(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        buffer_shape: [usize; 3],
        buffer_origin: [usize; 3],
        region: [usize; 3],
        dtype: DType,
        mtype: MType,
        data: &mut [u8],
    ) -> Result<()> {
        let _ = (
            device,
            mem,
            buffer_shape,
            buffer_origin,
            region,
            dtype,
            mtype,
            data,
        );
        Err(CleError::Other(
            "memory region read is not implemented for this backend".to_string(),
        ))
    }

    fn copy_memory_buffer_to_buffer(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        dst: &GpuMemPtr,
        byte_size: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn copy_memory_buffer_to_buffer_region(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        src_origin: [usize; 3],
        src_shape: [usize; 3],
        dst: &GpuMemPtr,
        _dst_origin: [usize; 3],
        dst_shape: [usize; 3],
        region: [usize; 3],
        item_size: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn copy_memory_buffer_to_image(
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
        Err(CleError::Other(
            "buffer-to-image copy is not implemented for this backend".to_string(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_memory_image_to_buffer(
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
        Err(CleError::Other(
            "image-to-buffer copy is not implemented for this backend".to_string(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_memory_image_to_image(
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
        Err(CleError::Other(
            "image-to-image copy is not implemented for this backend".to_string(),
        ))
    }

    fn set_memory(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        value: f32,
        dtype: DType,
        element_count: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn set_memory_region(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        buffer_shape: [usize; 3],
        buffer_origin: [usize; 3],
        region: [usize; 3],
        dtype: DType,
        mtype: MType,
        value: f32,
    ) -> Result<()> {
        match mtype {
            MType::Buffer => self.set_memory(
                device,
                mem,
                value,
                dtype,
                region[0] * region[1] * region[2],
            ),
            MType::Image => Err(CleError::Other(format!(
                "image fill is not implemented for this backend: shape={buffer_shape:?}, origin={buffer_origin:?}"
            ))),
        }
    }

    fn execute_kernel(
        &self,
        device: &DeviceArc,
        kernel_source: &str,
        kernel_name: &str,
        global_size: [usize; 3],
        local_size: [usize; 3],
        args: &[KernelArg],
    ) -> Result<()>;

    fn get_preamble(&self) -> Result<&'static str>;
}

// ── Kernel argument types ────────────────────────────────────────────────────

pub enum KernelArg {
    Mem(GpuMemPtr),
    Float(f32),
    Int(i32),
    Uint(u32),
    SizeT(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opencl_backend_type_alias_matches_clic_name() {
        let backend = OpenCLBackend;
        assert_eq!(backend.get_type(), "OpenCL");
    }

    #[test]
    fn image_channel_type_matches_clic_dtype_mapping() {
        assert_eq!(to_image_channel_type(DType::Float), CL_FLOAT);
        assert_eq!(to_image_channel_type(DType::Int32), CL_SIGNED_INT32);
        assert_eq!(to_image_channel_type(DType::Uint32), CL_UNSIGNED_INT32);
        assert_eq!(to_image_channel_type(DType::Int16), CL_SIGNED_INT16);
        assert_eq!(to_image_channel_type(DType::Uint16), CL_UNSIGNED_INT16);
        assert_eq!(to_image_channel_type(DType::Int8), CL_SIGNED_INT8);
        assert_eq!(to_image_channel_type(DType::Uint8), CL_UNSIGNED_INT8);
        assert_eq!(to_image_channel_type(DType::Unknown), CL_FLOAT);
    }

    #[test]
    fn buffer_pitches_match_clic_rect_rules() {
        let one_d = compute_buffer_pitches([16, 1, 1]);
        assert_eq!(one_d.row, 0);
        assert_eq!(one_d.slice, 0);

        let two_d = compute_buffer_pitches([16, 8, 1]);
        assert_eq!(two_d.row, 16);
        assert_eq!(two_d.slice, 0);

        let three_d = compute_buffer_pitches([16, 8, 4]);
        assert_eq!(three_d.row, 16);
        assert_eq!(three_d.slice, 128);
    }
}

// ── OpenCL backend implementation ────────────────────────────────────────────

pub struct OpenCLBackend;

impl OpenCLBackend {
    /// Canonical constructor alias for CLIc's `OpenCLBackend::OpenCLBackend()`.
    pub fn new() -> Self {
        Self
    }

    /// Canonical lazy-initialization alias for CLIc's `initialiseResources()`.
    pub fn initialise_resources(&self) -> Result<()> {
        Ok(())
    }

    fn cast_device(device: &DeviceArc) -> &OpenCLDevice {
        (device.as_ref() as &dyn Any)
            .downcast_ref::<OpenCLDevice>()
            .expect("DeviceArc does not contain an OpenCLDevice")
    }

    /// Canonical alias for CLIc's `OpenCLBackend::getType()`.
    pub fn get_type(&self) -> &'static str {
        "OpenCL"
    }

    /// Canonical alias for CLIc's `OpenCLBackend::getDevices()`.
    pub fn get_devices(&self, device_type: &str) -> Result<Vec<DeviceArc>> {
        if device_type != "gpu" && device_type != "cpu" {
            return crate::device::enumerate_opencl_devices("all");
        }
        crate::device::enumerate_opencl_devices(device_type)
    }

    /// Canonical alias for CLIc's `OpenCLBackend::getDevice()`.
    pub fn get_device(&self, name: &str, device_type: &str) -> Result<DeviceArc> {
        let devices_all = self.get_devices("all")?;
        if devices_all.is_empty() {
            eprintln!("Warning: Fail to find any OpenCL compatible devices.");
            return Err(CleError::NoDevicesFound);
        }
        let devices = self.get_devices(device_type)?;
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

    /// Canonical alias for CLIc's `OpenCLBackend::getDeviceFromIndex()`.
    pub fn get_device_from_index(&self, index: usize, device_type: &str) -> Result<DeviceArc> {
        let devices_all = self.get_devices("all")?;
        if devices_all.is_empty() {
            eprintln!("Warning: Fail to find any OpenCL compatible devices.");
            return Err(CleError::NoDevicesFound);
        }
        let devices = self.get_devices(device_type)?;
        if devices.is_empty() {
            return Ok(devices_all.last().unwrap().clone());
        }
        if index < devices.len() {
            return Ok(devices[index].clone());
        }
        Ok(devices.last().unwrap().clone())
    }

    /// Canonical alias for CLIc's `OpenCLBackend::getDevicesList()`.
    pub fn get_devices_list(&self, device_type: &str) -> Result<Vec<String>> {
        Ok(self
            .get_devices(device_type)?
            .into_iter()
            .map(|device| device.get_name().to_string())
            .collect())
    }

    /// Canonical buffer allocation alias for CLIc's `allocateBuffer()`.
    pub fn allocate_buffer(&self, device: &DeviceArc, byte_size: usize) -> Result<GpuMemPtr> {
        let ocl = Self::cast_device(device);
        let buf = unsafe {
            Buffer::<u8>::create(&ocl.context, CL_MEM_READ_WRITE, byte_size, ptr::null_mut())
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(wrap_mem_object(buf))
    }

    /// Canonical reference-count alias for CLIc's `getRefCount()`.
    pub fn get_ref_count(&self, mem: &GpuMemPtr) -> Result<usize> {
        match mem.as_ref() {
            GpuMemory::Buffer(mutex_buf) => {
                let guard = mutex_buf.lock().unwrap();
                guard
                    .reference_count()
                    .map(|count| count as usize)
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))
            }
            GpuMemory::Image(mutex_image) => {
                let guard = mutex_image.lock().unwrap();
                guard
                    .reference_count()
                    .map(|count| count as usize)
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))
            }
        }
    }

    /// Canonical free alias for CLIc's `freeMemory()`.
    pub fn free_memory(&self, _device: &DeviceArc, _mtype: MType, _mem: GpuMemPtr) {}

    /// Canonical buffer upload alias for CLIc's `writeBuffer()`.
    pub fn write_buffer(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        buffer_shape: [usize; 3],
        buffer_origin: [usize; 3],
        region: [usize; 3],
        host_data: &[u8],
    ) -> Result<()> {
        let opencl_device = Self::cast_device(device);
        let GpuMemory::Buffer(mutex_buf) = mem.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL buffer memory".to_string(),
            ));
        };
        let mut guard = mutex_buf.lock().unwrap();
        let pitches = compute_buffer_pitches(buffer_shape);
        let host_origin = [0, 0, 0];
        let _evt = unsafe {
            if buffer_shape[2] > 1 || buffer_shape[1] > 1 {
                opencl_device
                    .queue
                    .enqueue_write_buffer_rect(
                        &mut *guard,
                        CL_TRUE,
                        buffer_origin.as_ptr(),
                        host_origin.as_ptr(),
                        region.as_ptr(),
                        pitches.row,
                        pitches.slice,
                        0,
                        0,
                        host_data.as_ptr() as *mut c_void,
                        &[],
                    )
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
            } else {
                let host_ptr = std::slice::from_raw_parts(host_data.as_ptr(), region[0]);
                opencl_device
                    .queue
                    .enqueue_write_buffer(&mut *guard, CL_TRUE, buffer_origin[0], host_ptr, &[])
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
            }
        };
        Ok(())
    }

    /// Canonical buffer download alias for CLIc's `readBuffer()`.
    pub fn read_buffer(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        buffer_shape: [usize; 3],
        buffer_origin: [usize; 3],
        region: [usize; 3],
        host_data: &mut [u8],
    ) -> Result<()> {
        let opencl_device = Self::cast_device(device);
        let GpuMemory::Buffer(mutex_buf) = mem.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL buffer memory".to_string(),
            ));
        };
        let guard = mutex_buf.lock().unwrap();
        let pitches = compute_buffer_pitches(buffer_shape);
        let host_origin = [0, 0, 0];
        let _evt = unsafe {
            if buffer_shape[2] > 1 || buffer_shape[1] > 1 {
                opencl_device
                    .queue
                    .enqueue_read_buffer_rect(
                        &*guard,
                        CL_TRUE,
                        buffer_origin.as_ptr(),
                        host_origin.as_ptr(),
                        region.as_ptr(),
                        pitches.row,
                        pitches.slice,
                        0,
                        0,
                        host_data.as_mut_ptr() as *mut c_void,
                        &[],
                    )
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
            } else {
                let host_ptr = std::slice::from_raw_parts_mut(host_data.as_mut_ptr(), region[0]);
                opencl_device
                    .queue
                    .enqueue_read_buffer(&guard, CL_TRUE, buffer_origin[0], host_ptr, &[])
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
            }
        };
        Ok(())
    }

    /// Canonical fill alias for CLIc's `setBuffer()`.
    pub fn set_buffer(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        _buffer_shape: [usize; 3],
        _buffer_origin: [usize; 3],
        region: [usize; 3],
        dtype: DType,
        value: f32,
    ) -> Result<()> {
        let opencl_device = Self::cast_device(device);
        let size = region[0] * region[1] * region[2] * to_bytes(dtype);
        let GpuMemory::Buffer(mutex_buf) = mem.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL buffer memory".to_string(),
            ));
        };
        let mut guard = mutex_buf.lock().unwrap();

        match dtype {
            DType::Float => fill_buffer_typed(&opencl_device.queue, &mut *guard, size, value),
            DType::Int32 => {
                fill_buffer_typed(&opencl_device.queue, &mut *guard, size, value as i32)
            }
            DType::Uint32 => {
                fill_buffer_typed(&opencl_device.queue, &mut *guard, size, value as u32)
            }
            DType::Int16 => {
                fill_buffer_typed(&opencl_device.queue, &mut *guard, size, value as i16)
            }
            DType::Uint16 => {
                fill_buffer_typed(&opencl_device.queue, &mut *guard, size, value as u16)
            }
            DType::Int8 => fill_buffer_typed(&opencl_device.queue, &mut *guard, size, value as i8),
            DType::Uint8 => fill_buffer_typed(&opencl_device.queue, &mut *guard, size, value as u8),
            _ => Err(CleError::InvalidDtype),
        }
    }

    /// Canonical alias for CLIc's `buildKernel()`.
    pub fn build_kernel(
        &self,
        device: &DeviceArc,
        kernel_source: &str,
        kernel_name: &str,
    ) -> Result<Kernel> {
        let opencl_device = Self::cast_device(device);
        let disk_cache = crate::cache::DiskCache::instance();
        let source_hash = crate::cache::DiskCache::hash(kernel_source);
        let device_hash = crate::cache::DiskCache::hash(&opencl_device.get_info());
        let cache_key = format!("{}_{}", device_hash, source_hash);

        let mut program = device.get_program_from_cache(&cache_key);

        if program.is_none() && disk_cache.is_enabled() {
            program = load_program_from_cache(opencl_device, &device_hash, &source_hash);
        }

        if program.is_none() {
            let compiled = create_program_from_source(opencl_device, kernel_source)?;
            if disk_cache.is_enabled() {
                save_binary_to_cache(&device_hash, &source_hash, &compiled, device);
            }
            program = Some(compiled);
        }

        let program = program.expect("program is compiled or loaded");
        device.add_program_to_cache(cache_key, program.clone());

        Kernel::create(&program, kernel_name)
            .map(wrap_kernel)
            .map_err(|e| {
                CleError::OpenCL(format!("Kernel '{}' create failed: {:?}", kernel_name, e))
            })
    }

    /// Canonical image allocation alias for CLIc's `allocateImage()`.
    pub fn allocate_image(
        &self,
        device: &DeviceArc,
        region: [usize; 3],
        dtype: DType,
    ) -> Result<GpuMemPtr> {
        let ocl = Self::cast_device(device);

        let image_format = cl_image_format {
            image_channel_order: CL_R,
            image_channel_data_type: to_image_channel_type(dtype),
        };
        let image_desc = cl_image_desc {
            image_type: if region[2] > 1 {
                CL_MEM_OBJECT_IMAGE3D
            } else if region[1] > 1 {
                CL_MEM_OBJECT_IMAGE2D
            } else {
                CL_MEM_OBJECT_IMAGE1D
            },
            image_width: region[0],
            image_height: region[1],
            image_depth: region[2],
            image_array_size: 1,
            image_row_pitch: 0,
            image_slice_pitch: 0,
            num_mip_levels: 0,
            num_samples: 0,
            buffer: ptr::null_mut(),
        };
        let image = unsafe {
            Image::create(
                &ocl.context,
                CL_MEM_READ_WRITE,
                &image_format,
                &image_desc,
                ptr::null_mut(),
            )
            .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(Arc::new(GpuMemory::Image(Mutex::new(image))))
    }

    /// Canonical image upload alias for CLIc's `writeImage()`.
    pub fn write_image(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        _buffer_shape: [usize; 3],
        buffer_origin: [usize; 3],
        region: [usize; 3],
        host_data: &[u8],
    ) -> Result<()> {
        let ocl = Self::cast_device(device);
        let GpuMemory::Image(mutex_image) = mem.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL image memory".to_string(),
            ));
        };
        let mut guard = mutex_image.lock().unwrap();
        let _evt = unsafe {
            ocl.queue
                .enqueue_write_image(
                    &mut guard,
                    CL_TRUE,
                    buffer_origin.as_ptr(),
                    region.as_ptr(),
                    0,
                    0,
                    host_data.as_ptr() as *mut c_void,
                    &[],
                )
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    /// Canonical image download alias for CLIc's `readImage()`.
    pub fn read_image(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        _buffer_shape: [usize; 3],
        buffer_origin: [usize; 3],
        region: [usize; 3],
        host_data: &mut [u8],
    ) -> Result<()> {
        let ocl = Self::cast_device(device);
        let GpuMemory::Image(mutex_image) = mem.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL image memory".to_string(),
            ));
        };
        let guard = mutex_image.lock().unwrap();
        let _evt = unsafe {
            ocl.queue
                .enqueue_read_image(
                    &guard,
                    CL_TRUE,
                    buffer_origin.as_ptr(),
                    region.as_ptr(),
                    0,
                    0,
                    host_data.as_mut_ptr() as *mut c_void,
                    &[],
                )
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    /// Canonical alias for CLIc's `copyMemoryBufferToImage()`.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_memory_buffer_to_image(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        src_origin: [usize; 3],
        src_shape: [usize; 3],
        dst: &GpuMemPtr,
        dst_origin: [usize; 3],
        _dst_shape: [usize; 3],
        region: [usize; 3],
        _item_size: usize,
    ) -> Result<()> {
        let ocl = Self::cast_device(device);
        let GpuMemory::Buffer(src_mutex) = src.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL buffer memory".to_string(),
            ));
        };
        let GpuMemory::Image(dst_mutex) = dst.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL image memory".to_string(),
            ));
        };
        let src_guard = src_mutex.lock().unwrap();
        let mut dst_guard = dst_mutex.lock().unwrap();
        let src_pitches = compute_buffer_pitches(src_shape);
        let buffer_offset =
            src_origin[0] + src_origin[1] * src_pitches.row + src_origin[2] * src_pitches.slice;
        let _evt = unsafe {
            ocl.queue
                .enqueue_copy_buffer_to_image(
                    &src_guard,
                    &mut dst_guard,
                    buffer_offset,
                    dst_origin.as_ptr(),
                    region.as_ptr(),
                    &[],
                )
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    /// Canonical alias for CLIc's `copyMemoryImageToBuffer()`.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_memory_image_to_buffer(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        src_origin: [usize; 3],
        _src_shape: [usize; 3],
        dst: &GpuMemPtr,
        _dst_origin: [usize; 3],
        dst_shape: [usize; 3],
        region: [usize; 3],
        _item_size: usize,
    ) -> Result<()> {
        let ocl = Self::cast_device(device);
        let GpuMemory::Image(src_mutex) = src.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL image memory".to_string(),
            ));
        };
        let GpuMemory::Buffer(dst_mutex) = dst.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL buffer memory".to_string(),
            ));
        };
        let src_guard = src_mutex.lock().unwrap();
        let mut dst_guard = dst_mutex.lock().unwrap();
        let dst_pitches = compute_buffer_pitches(dst_shape);
        let buffer_offset =
            src_origin[0] + src_origin[1] * dst_pitches.row + src_origin[2] * dst_pitches.slice;
        let _evt = unsafe {
            ocl.queue
                .enqueue_copy_image_to_buffer(
                    &src_guard,
                    &mut dst_guard,
                    src_origin.as_ptr(),
                    region.as_ptr(),
                    buffer_offset,
                    &[],
                )
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    /// Canonical alias for CLIc's `copyMemoryImageToImage()`.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_memory_image_to_image(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        src_origin: [usize; 3],
        _src_shape: [usize; 3],
        dst: &GpuMemPtr,
        dst_origin: [usize; 3],
        _dst_shape: [usize; 3],
        region: [usize; 3],
        _item_size: usize,
    ) -> Result<()> {
        let ocl = Self::cast_device(device);
        let GpuMemory::Image(src_mutex) = src.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL image memory".to_string(),
            ));
        };
        let GpuMemory::Image(dst_mutex) = dst.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL image memory".to_string(),
            ));
        };
        let src_guard = src_mutex.lock().unwrap();
        let mut dst_guard = dst_mutex.lock().unwrap();
        let _evt = unsafe {
            ocl.queue
                .enqueue_copy_image(
                    &src_guard,
                    &mut dst_guard,
                    src_origin.as_ptr(),
                    dst_origin.as_ptr(),
                    region.as_ptr(),
                    &[],
                )
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    /// Canonical alias for CLIc's `setImage()`.
    pub fn set_image(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        _buffer_shape: [usize; 3],
        buffer_origin: [usize; 3],
        region: [usize; 3],
        dtype: DType,
        value: f32,
    ) -> Result<()> {
        let ocl = Self::cast_device(device);
        let GpuMemory::Image(mutex_image) = mem.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL image memory".to_string(),
            ));
        };
        let mut guard = mutex_image.lock().unwrap();

        match dtype {
            DType::Float => fill_image_typed(&ocl.queue, &mut guard, buffer_origin, region, value),
            DType::Int32 | DType::Int16 | DType::Int8 => {
                fill_image_typed(&ocl.queue, &mut guard, buffer_origin, region, value as i32)
            }
            DType::Uint32 | DType::Uint16 | DType::Uint8 => {
                fill_image_typed(&ocl.queue, &mut guard, buffer_origin, region, value as u32)
            }
            _ => Err(CleError::InvalidDtype),
        }
    }
}

impl Drop for OpenCLBackend {
    fn drop(&mut self) {}
}

impl Backend for OpenCLBackend {
    fn get_preamble(&self) -> Result<&'static str> {
        Ok(include_str!("../kernels/preamble.cl"))
    }

    fn allocate_memory(
        &self,
        device: &DeviceArc,
        region: [usize; 3],
        dtype: DType,
        mtype: MType,
    ) -> Result<GpuMemPtr> {
        match mtype {
            MType::Buffer => {
                let size = region[0] * region[1] * region[2] * to_bytes(dtype);
                self.allocate_buffer(device, size)
            }
            MType::Image => self.allocate_image(device, region, dtype),
        }
    }

    fn write_memory(&self, device: &DeviceArc, mem: &GpuMemPtr, data: &[u8]) -> Result<()> {
        let ocl = Self::cast_device(device);
        let GpuMemory::Buffer(mutex_buf) = mem.as_ref() else {
            return Err(CleError::Other(
                "Error: write_memory requires buffer memory; use write_memory_region for images"
                    .to_string(),
            ));
        };
        let mut guard = mutex_buf.lock().unwrap();
        let _evt = unsafe {
            ocl.queue
                .enqueue_write_buffer(&mut *guard, CL_TRUE, 0, data, &[])
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    fn read_memory(&self, device: &DeviceArc, mem: &GpuMemPtr, data: &mut [u8]) -> Result<()> {
        let ocl = Self::cast_device(device);
        let GpuMemory::Buffer(mutex_buf) = mem.as_ref() else {
            return Err(CleError::Other(
                "Error: read_memory requires buffer memory; use read_memory_region for images"
                    .to_string(),
            ));
        };
        let guard = mutex_buf.lock().unwrap();
        let _evt = unsafe {
            ocl.queue
                .enqueue_read_buffer(&guard, CL_TRUE, 0, data, &[])
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    fn write_memory_region(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        mut buffer_shape: [usize; 3],
        mut buffer_origin: [usize; 3],
        mut region: [usize; 3],
        dtype: DType,
        mtype: MType,
        data: &[u8],
    ) -> Result<()> {
        match mtype {
            MType::Buffer => {
                let elem_size = to_bytes(dtype);
                buffer_shape[0] *= elem_size;
                buffer_origin[0] *= elem_size;
                region[0] *= elem_size;
                self.write_buffer(device, mem, buffer_shape, buffer_origin, region, data)
            }
            MType::Image => {
                self.write_image(device, mem, buffer_shape, buffer_origin, region, data)
            }
        }
    }

    fn read_memory_region(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        mut buffer_shape: [usize; 3],
        mut buffer_origin: [usize; 3],
        mut region: [usize; 3],
        dtype: DType,
        mtype: MType,
        data: &mut [u8],
    ) -> Result<()> {
        match mtype {
            MType::Buffer => {
                let elem_size = to_bytes(dtype);
                buffer_shape[0] *= elem_size;
                buffer_origin[0] *= elem_size;
                region[0] *= elem_size;
                self.read_buffer(device, mem, buffer_shape, buffer_origin, region, data)
            }
            MType::Image => self.read_image(device, mem, buffer_shape, buffer_origin, region, data),
        }
    }

    fn copy_memory_buffer_to_buffer(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        dst: &GpuMemPtr,
        byte_size: usize,
    ) -> Result<()> {
        let ocl = Self::cast_device(device);
        let GpuMemory::Buffer(src_mutex) = src.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL buffer memory".to_string(),
            ));
        };
        let GpuMemory::Buffer(dst_mutex) = dst.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL buffer memory".to_string(),
            ));
        };
        let src_guard = src_mutex.lock().unwrap();
        let mut dst_guard = dst_mutex.lock().unwrap();
        let _evt = unsafe {
            ocl.queue
                .enqueue_copy_buffer(&*src_guard, &mut *dst_guard, 0, 0, byte_size, &[])
                .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
        };
        Ok(())
    }

    fn copy_memory_buffer_to_buffer_region(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        mut src_origin: [usize; 3],
        mut src_shape: [usize; 3],
        dst: &GpuMemPtr,
        mut dst_origin: [usize; 3],
        mut dst_shape: [usize; 3],
        mut region: [usize; 3],
        bytes: usize,
    ) -> Result<()> {
        let ocl = Self::cast_device(device);

        region[0] *= bytes;
        src_origin[0] *= bytes;
        src_shape[0] *= bytes;
        dst_origin[0] *= bytes;
        dst_shape[0] *= bytes;

        let src_pitches = compute_buffer_pitches(src_shape);
        let dst_pitches = compute_buffer_pitches(dst_shape);

        let GpuMemory::Buffer(src_mutex) = src.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL buffer memory".to_string(),
            ));
        };
        let GpuMemory::Buffer(dst_mutex) = dst.as_ref() else {
            return Err(CleError::Other(
                "Error: Expected OpenCL buffer memory".to_string(),
            ));
        };
        let src_guard = src_mutex.lock().unwrap();
        let mut dst_guard = dst_mutex.lock().unwrap();

        let _evt = unsafe {
            if dst_shape[2] > 1 || dst_shape[1] > 1 || src_shape[2] > 1 || src_shape[1] > 1 {
                ocl.queue
                    .enqueue_copy_buffer_rect(
                        &*src_guard,
                        &mut *dst_guard,
                        src_origin.as_ptr(),
                        dst_origin.as_ptr(),
                        region.as_ptr(),
                        src_pitches.row,
                        src_pitches.slice,
                        dst_pitches.row,
                        dst_pitches.slice,
                        &[],
                    )
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
            } else {
                ocl.queue
                    .enqueue_copy_buffer(
                        &*src_guard,
                        &mut *dst_guard,
                        src_origin[0],
                        dst_origin[0],
                        region[0],
                        &[],
                    )
                    .map_err(|e| CleError::OpenCL(format!("{:?}", e)))?
            }
        };
        Ok(())
    }

    fn copy_memory_buffer_to_image(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        src_origin: [usize; 3],
        src_shape: [usize; 3],
        dst: &GpuMemPtr,
        dst_origin: [usize; 3],
        dst_shape: [usize; 3],
        region: [usize; 3],
        item_size: usize,
    ) -> Result<()> {
        OpenCLBackend::copy_memory_buffer_to_image(
            self, device, src, src_origin, src_shape, dst, dst_origin, dst_shape, region, item_size,
        )
    }

    fn copy_memory_image_to_buffer(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        src_origin: [usize; 3],
        src_shape: [usize; 3],
        dst: &GpuMemPtr,
        dst_origin: [usize; 3],
        dst_shape: [usize; 3],
        region: [usize; 3],
        item_size: usize,
    ) -> Result<()> {
        OpenCLBackend::copy_memory_image_to_buffer(
            self, device, src, src_origin, src_shape, dst, dst_origin, dst_shape, region, item_size,
        )
    }

    fn copy_memory_image_to_image(
        &self,
        device: &DeviceArc,
        src: &GpuMemPtr,
        src_origin: [usize; 3],
        src_shape: [usize; 3],
        dst: &GpuMemPtr,
        dst_origin: [usize; 3],
        dst_shape: [usize; 3],
        region: [usize; 3],
        item_size: usize,
    ) -> Result<()> {
        OpenCLBackend::copy_memory_image_to_image(
            self, device, src, src_origin, src_shape, dst, dst_origin, dst_shape, region, item_size,
        )
    }

    fn set_memory(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        value: f32,
        dtype: DType,
        element_count: usize,
    ) -> Result<()> {
        match mem.as_ref() {
            GpuMemory::Buffer(_) => self.set_buffer(
                device,
                mem,
                [element_count, 1, 1],
                [0, 0, 0],
                [element_count, 1, 1],
                dtype,
                value,
            ),
            GpuMemory::Image(_) => self.set_image(
                device,
                mem,
                [element_count, 1, 1],
                [0, 0, 0],
                [element_count, 1, 1],
                dtype,
                value,
            ),
        }
    }

    fn set_memory_region(
        &self,
        device: &DeviceArc,
        mem: &GpuMemPtr,
        buffer_shape: [usize; 3],
        buffer_origin: [usize; 3],
        region: [usize; 3],
        dtype: DType,
        mtype: MType,
        value: f32,
    ) -> Result<()> {
        match mtype {
            MType::Buffer => self.set_buffer(
                device,
                mem,
                buffer_shape,
                buffer_origin,
                region,
                dtype,
                value,
            ),
            MType::Image => self.set_image(
                device,
                mem,
                buffer_shape,
                buffer_origin,
                region,
                dtype,
                value,
            ),
        }
    }

    fn execute_kernel(
        &self,
        device: &DeviceArc,
        kernel_source: &str,
        kernel_name: &str,
        global_size: [usize; 3],
        local_size: [usize; 3],
        args: &[KernelArg],
    ) -> Result<()> {
        let opencl_device = Self::cast_device(device);
        let kernel = self.build_kernel(device, kernel_source, kernel_name)?;

        // Set kernel arguments
        for (i, arg) in args.iter().enumerate() {
            match arg {
                KernelArg::Mem(mem) => {
                    let handle: cl_mem = match mem.as_ref() {
                        GpuMemory::Buffer(m) => {
                            let guard = m.lock().unwrap();
                            guard.get()
                        }
                        GpuMemory::Image(m) => {
                            let guard = m.lock().unwrap();
                            guard.get()
                        }
                    };
                    unsafe {
                        kernel.set_arg(i as u32, &handle).map_err(|e| {
                            CleError::OpenCL(format!("set_arg({}) mem failed: {:?}", i, e))
                        })?
                    };
                }
                KernelArg::Float(v) => {
                    unsafe {
                        kernel.set_arg(i as u32, v).map_err(|e| {
                            CleError::OpenCL(format!("set_arg({}) float failed: {:?}", i, e))
                        })?
                    };
                }
                KernelArg::Int(v) => {
                    unsafe {
                        kernel.set_arg(i as u32, v).map_err(|e| {
                            CleError::OpenCL(format!("set_arg({}) int failed: {:?}", i, e))
                        })?
                    };
                }
                KernelArg::Uint(v) => {
                    unsafe {
                        kernel.set_arg(i as u32, v).map_err(|e| {
                            CleError::OpenCL(format!("set_arg({}) uint failed: {:?}", i, e))
                        })?
                    };
                }
                KernelArg::SizeT(v) => {
                    unsafe {
                        kernel.set_arg(i as u32, v).map_err(|e| {
                            CleError::OpenCL(format!("set_arg({}) sizet failed: {:?}", i, e))
                        })?
                    };
                }
            }
        }

        let local_ptr = if local_size[0] == 0 {
            ptr::null()
        } else {
            local_size.as_ptr()
        };

        let _evt = unsafe {
            opencl_device
                .queue
                .enqueue_nd_range_kernel(
                    kernel.get(),
                    3,
                    ptr::null(),
                    global_size.as_ptr(),
                    local_ptr,
                    &[],
                )
                .map_err(|e| {
                    let mut msg = format!(
                        "Error: Failed to launch kernel '{}'. OpenCL error: {:?}",
                        kernel_name, e
                    );
                    msg.push_str(&format!(
                        "\nGlobal work size: [{}, {}, {}]",
                        global_size[0], global_size[1], global_size[2]
                    ));
                    if local_size[0] > 0 {
                        msg.push_str(&format!(
                            "\nLocal work size: [{}, {}, {}]",
                            local_size[0], local_size[1], local_size[2]
                        ));
                    }
                    CleError::OpenCL(msg)
                })?
        };

        device.finish();

        Ok(())
    }
}
