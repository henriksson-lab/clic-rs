use std::fmt;
use std::sync::{Arc, Mutex};

use crate::backend::GpuMemPtr;
use crate::backend_manager::BackendManager;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::types::{
    to_bytes, to_mtype_string, to_string as dtype_to_string, to_type, DType, GpuScalar, MType,
};
use crate::utils::shape_to_dimension;

/// GPU-resident n-dimensional array.
pub struct Array {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) depth: usize,
    /// Explicit dimensionality passed at creation (may differ from shape_to_dimension).
    pub(crate) dim: usize,
    pub(crate) dtype: DType,
    pub(crate) mtype: MType,
    pub(crate) device: DeviceArc,
    pub(crate) mem: Option<GpuMemPtr>,
    pub(crate) owns_memory: bool,
}

/// Shared, mutable GPU array handle — `Arc<Mutex<Array>>`.
pub type ArrayPtr = Arc<Mutex<Array>>;

impl Array {
    // ── Constructors ─────────────────────────────────────────────────────────

    fn new(
        width: usize,
        height: usize,
        depth: usize,
        dim: usize,
        dtype: DType,
        mtype: MType,
        device: &DeviceArc,
    ) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            depth: depth.max(1),
            dim,
            dtype,
            mtype,
            device: device.clone(),
            mem: None,
            owns_memory: true,
        }
    }

    fn from_gpu_memory(
        width: usize,
        height: usize,
        depth: usize,
        dim: usize,
        dtype: DType,
        mtype: MType,
        mem: GpuMemPtr,
        device: &DeviceArc,
    ) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            depth: depth.max(1),
            dim,
            dtype,
            mtype,
            device: device.clone(),
            mem: Some(mem),
            owns_memory: false,
        }
    }

    /// Allocate a new GPU array.
    pub fn create(
        width: usize,
        height: usize,
        depth: usize,
        dim: usize,
        dtype: DType,
        mtype: MType,
        device: &DeviceArc,
    ) -> Result<ArrayPtr> {
        let mut a = Self::new(width, height, depth, dim, dtype, mtype, device);
        a.allocate()?;
        Ok(Arc::new(Mutex::new(a)))
    }

    /// Allocate and immediately upload host data.
    pub fn create_with_data<T: GpuScalar>(
        width: usize,
        height: usize,
        depth: usize,
        dim: usize,
        mtype: MType,
        data: &[T],
        device: &DeviceArc,
    ) -> Result<ArrayPtr> {
        let ptr = Self::create(width, height, depth, dim, to_type::<T>(), mtype, device)?;
        ptr.lock().unwrap().write_from(data)?;
        Ok(ptr)
    }

    /// Wrap an existing GPU allocation. Mirrors CLIc's `createFromGPUMemory()`.
    pub fn create_from_gpu_memory(
        width: usize,
        height: usize,
        depth: usize,
        dim: usize,
        dtype: DType,
        mtype: MType,
        mem: GpuMemPtr,
        device: &DeviceArc,
    ) -> Result<ArrayPtr> {
        Ok(Arc::new(Mutex::new(Self::from_gpu_memory(
            width, height, depth, dim, dtype, mtype, mem, device,
        ))))
    }

    /// Create an array with the same shape, dtype, memory type, and device as
    /// `src`. Mirrors CLIc's `Array::create(const Array::Pointer&)` overload.
    pub fn create_from_array(src: &ArrayPtr) -> Result<ArrayPtr> {
        let s = src.lock().unwrap();
        Self::create(
            s.width, s.height, s.depth, s.dim, s.dtype, s.mtype, &s.device,
        )
    }

    // ── Memory management ─────────────────────────────────────────────────────

    pub fn allocate(&mut self) -> Result<()> {
        if self.mem.is_some() {
            return Ok(());
        }
        let mem = BackendManager::get_instance().backend().allocate_memory(
            &self.device,
            [self.width, self.height, self.depth],
            self.dtype,
            self.mtype,
        )?;
        self.mem = Some(mem);
        Ok(())
    }

    // ── Data transfer ─────────────────────────────────────────────────────────

    /// Write typed host data into the full array. Mirrors CLIc's `writeFrom()`.
    pub fn write_from<T: GpuScalar>(&self, data: &[T]) -> Result<()> {
        // Safety: &[T] where T: Copy can be viewed as &[u8] for the purpose of GPU upload.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        if bytes.len() != self.bitsize() {
            return Err(CleError::DimensionMismatch);
        }
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        let origin = [0, 0, 0];
        let shape = [self.width, self.height, self.depth];
        let region = [self.width, self.height, self.depth];
        BackendManager::get_instance()
            .backend()
            .write_memory_region(
                &self.device,
                mem,
                shape,
                origin,
                region,
                self.dtype,
                self.mtype,
                bytes,
            )
    }

    /// Write typed host data into a rectangular region. Mirrors CLIc's region
    /// `writeFrom()` overload.
    pub fn write_from_region<T: GpuScalar>(
        &self,
        data: &[T],
        region: [usize; 3],
        origin: [usize; 3],
    ) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        if region.contains(&0) {
            return Err(CleError::Other(
                "array region must be non-empty".to_string(),
            ));
        }
        if origin[0] + region[0] > self.width
            || origin[1] + region[1] > self.height
            || origin[2] + region[2] > self.depth
        {
            return Err(CleError::DimensionMismatch);
        }
        let item_size = to_bytes(self.dtype);
        let expected = region[0] * region[1] * region[2] * item_size;
        if bytes.len() != expected {
            return Err(CleError::Other(format!(
                "write_from_region expected {expected} bytes, got {}",
                bytes.len()
            )));
        }

        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        BackendManager::get_instance()
            .backend()
            .write_memory_region(
                &self.device,
                mem,
                [self.width, self.height, self.depth],
                origin,
                region,
                self.dtype,
                self.mtype,
                bytes,
            )
    }

    /// Write one typed element at `(x, y, z)`. Mirrors CLIc's point
    /// `writeFrom(host_data, x, y, z)` overload.
    pub fn write_from_at<T: GpuScalar>(
        &self,
        data: &[T],
        x: usize,
        y: usize,
        z: usize,
    ) -> Result<()> {
        self.write_from_region(data, [1, 1, 1], [x, y, z])
    }

    /// Read the full array into typed host memory. Mirrors CLIc's `readTo()`.
    pub fn read_to<T: GpuScalar>(&self, data: &mut [T]) -> Result<()> {
        // Safety: &mut [T] where T: Copy can be viewed as &mut [u8] for GPU readback.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(data),
            )
        };
        if bytes.len() != self.bitsize() {
            return Err(CleError::DimensionMismatch);
        }
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        let origin = [0, 0, 0];
        let shape = [self.width, self.height, self.depth];
        let region = [self.width, self.height, self.depth];
        BackendManager::get_instance().backend().read_memory_region(
            &self.device,
            mem,
            shape,
            origin,
            region,
            self.dtype,
            self.mtype,
            bytes,
        )
    }

    /// Read a rectangular region into typed host memory. Mirrors CLIc's region
    /// `readTo()` overload.
    pub fn read_to_region<T: GpuScalar>(
        &self,
        data: &mut [T],
        region: [usize; 3],
        origin: [usize; 3],
    ) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(data),
            )
        };
        if region.contains(&0) {
            return Err(CleError::Other(
                "array region must be non-empty".to_string(),
            ));
        }
        if origin[0] + region[0] > self.width
            || origin[1] + region[1] > self.height
            || origin[2] + region[2] > self.depth
        {
            return Err(CleError::DimensionMismatch);
        }
        let item_size = to_bytes(self.dtype);
        let expected = region[0] * region[1] * region[2] * item_size;
        if bytes.len() != expected {
            return Err(CleError::Other(format!(
                "read_to_region expected {expected} bytes, got {}",
                bytes.len()
            )));
        }

        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        BackendManager::get_instance().backend().read_memory_region(
            &self.device,
            mem,
            [self.width, self.height, self.depth],
            origin,
            region,
            self.dtype,
            self.mtype,
            bytes,
        )
    }

    /// Read one typed element at `(x, y, z)`. Mirrors CLIc's point
    /// `readTo(host_data, x, y, z)` overload.
    pub fn read_to_at<T: GpuScalar>(
        &self,
        data: &mut [T],
        x: usize,
        y: usize,
        z: usize,
    ) -> Result<()> {
        self.read_to_region(data, [1, 1, 1], [x, y, z])
    }

    pub fn copy_to(&self, dst: &ArrayPtr) -> Result<()> {
        let dst_lock = dst.lock().unwrap();
        if !Arc::ptr_eq(&self.device, dst_lock.device()) {
            return Err(CleError::Other(
                "Error: Copying Arrays from different devices".to_string(),
            ));
        }
        if self.width != dst_lock.width
            || self.height != dst_lock.height
            || self.depth != dst_lock.depth
            || self.item_size() != dst_lock.item_size()
        {
            return Err(CleError::DimensionMismatch);
        }
        let src_origin = [0, 0, 0];
        let dst_origin = [0, 0, 0];
        let region = [self.width, self.height, self.depth];
        let src_shape = [self.width, self.height, self.depth];
        let dst_shape = [dst_lock.width, dst_lock.height, dst_lock.depth];

        let dst_ptr = dst_lock.mem.as_ref().ok_or(CleError::NotAllocated)?;

        if self.mtype == MType::Buffer && dst_lock.mtype == MType::Buffer {
            let src_mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
            BackendManager::get_instance()
                .backend()
                .copy_memory_buffer_to_buffer_region(
                    &self.device,
                    src_mem,
                    src_origin,
                    src_shape,
                    dst_ptr,
                    dst_origin,
                    dst_shape,
                    region,
                    to_bytes(self.dtype),
                )
        } else if self.mtype == MType::Image && dst_lock.mtype == MType::Image {
            let src_mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
            BackendManager::get_instance()
                .backend()
                .copy_memory_image_to_image(
                    &self.device,
                    src_mem,
                    src_origin,
                    src_shape,
                    dst_ptr,
                    dst_origin,
                    dst_shape,
                    region,
                    to_bytes(self.dtype),
                )
        } else if self.mtype == MType::Buffer && dst_lock.mtype == MType::Image {
            let src_mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
            BackendManager::get_instance()
                .backend()
                .copy_memory_buffer_to_image(
                    &self.device,
                    src_mem,
                    src_origin,
                    src_shape,
                    dst_ptr,
                    dst_origin,
                    dst_shape,
                    region,
                    to_bytes(self.dtype),
                )
        } else if self.mtype == MType::Image && dst_lock.mtype == MType::Buffer {
            let src_mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
            BackendManager::get_instance()
                .backend()
                .copy_memory_image_to_buffer(
                    &self.device,
                    src_mem,
                    src_origin,
                    src_shape,
                    dst_ptr,
                    dst_origin,
                    dst_shape,
                    region,
                    to_bytes(self.dtype),
                )
        } else {
            Err(CleError::Other(
                "Error: Copying Arrays from different memory types".to_string(),
            ))
        }
    }

    /// Copy a rectangular region into another array. Mirrors CLIc's region
    /// `copyTo()` overload for buffer-backed arrays.
    pub fn copy_to_region(
        &self,
        dst: &ArrayPtr,
        region: [usize; 3],
        src_origin: [usize; 3],
        dst_origin: [usize; 3],
    ) -> Result<()> {
        let dst_lock = dst.lock().unwrap();
        if !Arc::ptr_eq(&self.device, dst_lock.device()) {
            return Err(CleError::Other(
                "Error: Copying Arrays from different devices".to_string(),
            ));
        }
        let src_origin = src_origin;
        let dst_origin = dst_origin;
        let region = region;
        let src_shape = [self.width, self.height, self.depth];
        let dst_shape = [dst_lock.width, dst_lock.height, dst_lock.depth];

        let dst_ptr = dst_lock.mem.as_ref().ok_or(CleError::NotAllocated)?;

        if self.mtype == MType::Buffer && dst_lock.mtype == MType::Buffer {
            let src_mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
            BackendManager::get_instance()
                .backend()
                .copy_memory_buffer_to_buffer_region(
                    &self.device,
                    src_mem,
                    src_origin,
                    src_shape,
                    dst_ptr,
                    dst_origin,
                    dst_shape,
                    region,
                    to_bytes(self.dtype),
                )
        } else if self.mtype == MType::Image && dst_lock.mtype == MType::Image {
            let src_mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
            BackendManager::get_instance()
                .backend()
                .copy_memory_image_to_image(
                    &self.device,
                    src_mem,
                    src_origin,
                    src_shape,
                    dst_ptr,
                    dst_origin,
                    dst_shape,
                    region,
                    to_bytes(self.dtype),
                )
        } else if self.mtype == MType::Buffer && dst_lock.mtype == MType::Image {
            let src_mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
            BackendManager::get_instance()
                .backend()
                .copy_memory_buffer_to_image(
                    &self.device,
                    src_mem,
                    src_origin,
                    src_shape,
                    dst_ptr,
                    dst_origin,
                    dst_shape,
                    region,
                    to_bytes(self.dtype),
                )
        } else if self.mtype == MType::Image && dst_lock.mtype == MType::Buffer {
            let src_mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
            BackendManager::get_instance()
                .backend()
                .copy_memory_image_to_buffer(
                    &self.device,
                    src_mem,
                    src_origin,
                    src_shape,
                    dst_ptr,
                    dst_origin,
                    dst_shape,
                    region,
                    to_bytes(self.dtype),
                )
        } else {
            Ok(())
        }
    }

    pub fn fill(&self, value: f32) -> Result<()> {
        let origin = [0, 0, 0];
        let region = [self.width, self.height, self.depth];
        let shape = [self.width, self.height, self.depth];
        match self.dtype {
            DType::Float
            | DType::Int8
            | DType::Uint8
            | DType::Int16
            | DType::Uint16
            | DType::Int32
            | DType::Uint32 => {}
            DType::Complex | DType::Unknown => return Err(CleError::InvalidDtype),
        }
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        BackendManager::get_instance().backend().set_memory_region(
            &self.device,
            mem,
            shape,
            origin,
            region,
            self.dtype,
            self.mtype,
            value,
        )
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn dim(&self) -> usize {
        shape_to_dimension(self.width, self.height, self.depth)
    }
    /// Explicit dimensionality passed at creation. Mirrors CLIc's `dimension()`.
    pub fn dimension(&self) -> usize {
        self.dim
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn mtype(&self) -> MType {
        self.mtype
    }
    pub fn device(&self) -> &DeviceArc {
        &self.device
    }
    pub fn size(&self) -> usize {
        self.width * self.height * self.depth
    }
    /// Total byte size. Mirrors CLIc's `bitsize()`.
    pub fn bitsize(&self) -> usize {
        self.size() * self.item_size()
    }
    /// Size in bytes of one array item. Mirrors CLIc's `itemSize()`.
    pub fn item_size(&self) -> usize {
        to_bytes(self.dtype)
    }
    /// Whether device memory is initialized. Mirrors CLIc's `initialized()`.
    pub fn initialized(&self) -> bool {
        self.mem.is_some()
    }
    /// Whether this array owns its device allocation. Mirrors CLIc's `ownsMemory()`.
    pub fn owns_memory(&self) -> bool {
        self.owns_memory
    }

    /// Return the raw GPU memory pointer. Mirrors CLIc's mutable `get()`.
    pub fn get(&self) -> Option<&GpuMemPtr> {
        self.mem.as_ref()
    }
    /// Return the raw GPU memory pointer. Mirrors CLIc's const `c_get()`.
    pub fn c_get(&self) -> Option<&GpuMemPtr> {
        self.mem.as_ref()
    }
    /// Return the shared GPU memory handle. Mirrors CLIc's `get_ptr()`.
    pub fn get_ptr(&self) -> Option<GpuMemPtr> {
        self.mem.clone()
    }

    /// Validate an optional shared array handle. Mirrors CLIc's `Array::check_ptr()`.
    pub fn check_ptr(ptr: Option<&ArrayPtr>, error_message: &str) -> Result<()> {
        if ptr.is_none() {
            return Err(CleError::Other(error_message.to_string()));
        }
        Ok(())
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}dArray ([{},{},{}], dtype={}, mtype={})",
            self.dimension(),
            self.width,
            self.height,
            self.depth,
            dtype_to_string(self.dtype),
            to_mtype_string(self.mtype)
        )
    }
}

/// Print a typed array for debugging. Mirrors CLIc's templated `print()` helper.
pub fn print<T>(array: Option<&ArrayPtr>, name: &str) -> Result<()>
where
    T: GpuScalar + Default + fmt::Display,
{
    let Some(array) = array else {
        println!("Print Array::Pointer (nullptr)");
        return Ok(());
    };

    let lock = array.lock().unwrap();
    let mut host_data = vec![T::default(); lock.size()];
    lock.read_to(&mut host_data)?;
    let width = lock.width;
    let height = lock.height;
    let depth = lock.depth;
    drop(lock);

    println!("{name}:");
    for z in 0..depth {
        if depth > 1 {
            println!("z = {z}");
        }
        for y in 0..height {
            for x in 0..width {
                let index = z * height * width + y * width + x;
                print!("{} ", host_data[index]);
            }
            println!();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencl3::program::Program;

    struct DummyDevice;

    impl crate::device::Device for DummyDevice {
        fn get_name(&self) -> &str {
            "dummy"
        }
        fn get_device_type(&self) -> &str {
            "dummy"
        }
        fn support_image(&self) -> bool {
            false
        }
        fn get_maximum_buffer_size(&self) -> usize {
            0
        }
        fn get_maximum_work_group_size(&self) -> usize {
            1
        }
        fn get_local_memory_size(&self) -> usize {
            usize::MAX
        }
        fn get_platform(&self) -> String {
            String::new()
        }
        fn finish(&self) {}
        fn get_program_from_cache(&self, _key: &str) -> Option<Arc<Program>> {
            None
        }
        fn add_program_to_cache(&self, _key: String, _program: Arc<Program>) {}
        fn device_hash(&self) -> String {
            "dummy".to_string()
        }
    }

    fn unallocated_array(width: usize, height: usize, depth: usize) -> Array {
        unallocated_array_on(Arc::new(DummyDevice), width, height, depth)
    }

    fn unallocated_array_on(device: DeviceArc, width: usize, height: usize, depth: usize) -> Array {
        Array {
            width,
            height,
            depth,
            dim: shape_to_dimension(width, height, depth),
            dtype: DType::Float,
            mtype: MType::Buffer,
            device,
            mem: None,
            owns_memory: true,
        }
    }

    fn unallocated_array_with_type(dtype: DType, mtype: MType) -> Array {
        Array {
            width: 1,
            height: 1,
            depth: 1,
            dim: 1,
            dtype,
            mtype,
            device: Arc::new(DummyDevice),
            mem: None,
            owns_memory: true,
        }
    }

    #[test]
    fn full_write_rejects_wrong_element_count_before_backend() {
        let arr = unallocated_array(2, 2, 1);
        let err = arr.write_from(&[1.0_f32, 2.0]).unwrap_err();
        assert!(matches!(err, CleError::DimensionMismatch));
    }

    #[test]
    fn full_read_rejects_wrong_element_count_before_backend() {
        let arr = unallocated_array(2, 2, 1);
        let mut data = vec![0.0_f32; 2];
        let err = arr.read_to(&mut data).unwrap_err();
        assert!(matches!(err, CleError::DimensionMismatch));
    }

    #[test]
    fn full_copy_rejects_shape_mismatch_before_backend() {
        let device: DeviceArc = Arc::new(DummyDevice);
        let src = unallocated_array_on(device.clone(), 2, 2, 1);
        let dst = Arc::new(Mutex::new(unallocated_array_on(device, 4, 1, 1)));
        let err = src.copy_to(&dst).unwrap_err();
        assert!(matches!(err, CleError::DimensionMismatch));
    }

    #[test]
    fn fill_accepts_image_path_until_allocation_check() {
        let arr = unallocated_array_with_type(DType::Float, MType::Image);
        let err = arr.fill(1.0).unwrap_err();
        assert!(matches!(err, CleError::NotAllocated));
    }

    #[test]
    fn fill_rejects_unknown_dtype_before_backend() {
        let arr = unallocated_array_with_type(DType::Unknown, MType::Buffer);
        let err = arr.fill(1.0).unwrap_err();
        assert!(matches!(err, CleError::InvalidDtype));
    }
}
