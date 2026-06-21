use std::sync::{Arc, Mutex};

use crate::backend::GpuMemPtr;
use crate::backend_manager::BackendManager;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::types::{DType, GpuScalar, MType};
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
        let mut a = Array {
            width: width.max(1),
            height: height.max(1),
            depth: depth.max(1),
            dim,
            dtype,
            mtype,
            device: device.clone(),
            mem: None,
            owns_memory: true,
        };
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
        let ptr = Self::create(width, height, depth, dim, T::dtype(), mtype, device)?;
        ptr.lock().unwrap().write_from_typed(data)?;
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
        Ok(Arc::new(Mutex::new(Array {
            width: width.max(1),
            height: height.max(1),
            depth: depth.max(1),
            dim,
            dtype,
            mtype,
            device: device.clone(),
            mem: Some(mem),
            owns_memory: false,
        })))
    }

    /// Create an array with the same shape, dtype and mtype as `src`.
    pub fn create_like(src: &ArrayPtr, device: &DeviceArc) -> Result<ArrayPtr> {
        let s = src.lock().unwrap();
        Self::create(s.width, s.height, s.depth, s.dim, s.dtype, s.mtype, device)
    }

    /// Create an array like `src` but with a different dtype.
    pub fn create_like_typed(src: &ArrayPtr, dtype: DType, device: &DeviceArc) -> Result<ArrayPtr> {
        let s = src.lock().unwrap();
        Self::create(s.width, s.height, s.depth, s.dim, dtype, s.mtype, device)
    }

    // ── Memory management ─────────────────────────────────────────────────────

    pub fn allocate(&mut self) -> Result<()> {
        if self.mem.is_some() {
            return Ok(());
        }
        let mem = BackendManager::get().backend().allocate_memory(
            &self.device,
            [self.width, self.height, self.depth],
            self.dtype,
            self.mtype,
        )?;
        self.mem = Some(mem);
        Ok(())
    }

    // ── Data transfer ─────────────────────────────────────────────────────────

    pub fn write_from_typed<T: GpuScalar>(&self, data: &[T]) -> Result<()> {
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        // Safety: &[T] where T: Copy can be viewed as &[u8] for the purpose of GPU upload.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        BackendManager::get()
            .backend()
            .write_memory(&self.device, mem, bytes)
    }

    /// Write typed host data into the full array. Mirrors CLIc's `writeFrom()`.
    pub fn write_from<T: GpuScalar>(&self, data: &[T]) -> Result<()> {
        self.write_from_typed(data)
    }

    pub fn write_from_bytes(&self, data: &[u8]) -> Result<()> {
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        BackendManager::get()
            .backend()
            .write_memory(&self.device, mem, data)
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
        self.write_from_bytes_region(bytes, region, origin)
    }

    /// Write raw bytes into a rectangular region.
    pub fn write_from_bytes_region(
        &self,
        data: &[u8],
        region: [usize; 3],
        origin: [usize; 3],
    ) -> Result<()> {
        self.validate_region(region, origin)?;
        let item_size = self.dtype.byte_size();
        let expected = region[0] * region[1] * region[2] * item_size;
        if data.len() != expected {
            return Err(CleError::Other(format!(
                "write_from_region expected {expected} bytes, got {}",
                data.len()
            )));
        }

        let mut full = vec![0u8; self.byte_size()];
        self.read_to_bytes(&mut full)?;
        copy_region_bytes(
            data,
            [region[0], region[1], region[2]],
            &mut full,
            [self.width, self.height, self.depth],
            [0, 0, 0],
            origin,
            region,
            item_size,
        );
        self.write_from_bytes(&full)
    }

    pub fn read_to_typed<T: GpuScalar>(&self, data: &mut [T]) -> Result<()> {
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        // Safety: &mut [T] where T: Copy can be viewed as &mut [u8] for GPU readback.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(data),
            )
        };
        BackendManager::get()
            .backend()
            .read_memory(&self.device, mem, bytes)
    }

    /// Read the full array into typed host memory. Mirrors CLIc's `readTo()`.
    pub fn read_to<T: GpuScalar>(&self, data: &mut [T]) -> Result<()> {
        self.read_to_typed(data)
    }

    pub fn read_to_bytes(&self, data: &mut [u8]) -> Result<()> {
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        BackendManager::get()
            .backend()
            .read_memory(&self.device, mem, data)
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
        self.read_to_bytes_region(bytes, region, origin)
    }

    /// Read a rectangular region into raw host bytes.
    pub fn read_to_bytes_region(
        &self,
        data: &mut [u8],
        region: [usize; 3],
        origin: [usize; 3],
    ) -> Result<()> {
        self.validate_region(region, origin)?;
        let item_size = self.dtype.byte_size();
        let expected = region[0] * region[1] * region[2] * item_size;
        if data.len() != expected {
            return Err(CleError::Other(format!(
                "read_to_region expected {expected} bytes, got {}",
                data.len()
            )));
        }

        let mut full = vec![0u8; self.byte_size()];
        self.read_to_bytes(&mut full)?;
        copy_region_bytes(
            &full,
            [self.width, self.height, self.depth],
            data,
            [region[0], region[1], region[2]],
            origin,
            [0, 0, 0],
            region,
            item_size,
        );
        Ok(())
    }

    pub fn copy_to(&self, dst: &ArrayPtr) -> Result<()> {
        let src_mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        let dst_lock = dst.lock().unwrap();
        let dst_mem = dst_lock.mem.as_ref().ok_or(CleError::NotAllocated)?;
        let byte_size = self.size() * self.dtype.byte_size();
        BackendManager::get()
            .backend()
            .copy_memory(&self.device, src_mem, dst_mem, byte_size)
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
        self.validate_region(region, src_origin)?;
        let dst_lock = dst.lock().unwrap();
        dst_lock.validate_region(region, dst_origin)?;
        if self.dtype != dst_lock.dtype {
            return Err(CleError::InvalidDtype);
        }
        if self.mtype != dst_lock.mtype {
            return Err(CleError::Other(
                "copy_to_region: memory types do not match".to_string(),
            ));
        }

        let item_size = self.dtype.byte_size();
        let mut src_bytes = vec![0u8; self.byte_size()];
        let mut dst_bytes = vec![0u8; dst_lock.byte_size()];
        self.read_to_bytes(&mut src_bytes)?;
        dst_lock.read_to_bytes(&mut dst_bytes)?;
        copy_region_bytes(
            &src_bytes,
            [self.width, self.height, self.depth],
            &mut dst_bytes,
            [dst_lock.width, dst_lock.height, dst_lock.depth],
            src_origin,
            dst_origin,
            region,
            item_size,
        );
        dst_lock.write_from_bytes(&dst_bytes)
    }

    pub fn fill(&self, value: f32) -> Result<()> {
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        BackendManager::get().backend().set_memory(
            &self.device,
            mem,
            value,
            self.dtype,
            self.size(),
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
        self.dim
    }
    /// Effective dimensionality derived from shape (may differ from `dim()`).
    pub fn dimension(&self) -> usize {
        shape_to_dimension(self.width, self.height, self.depth)
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
    pub fn byte_size(&self) -> usize {
        self.size() * self.dtype.byte_size()
    }
    /// Total byte size. Mirrors CLIc's `bitsize()`.
    pub fn bitsize(&self) -> usize {
        self.byte_size()
    }
    /// Size in bytes of one array item. Mirrors CLIc's `itemSize()`.
    pub fn item_size(&self) -> usize {
        self.dtype.byte_size()
    }
    pub fn is_allocated(&self) -> bool {
        self.mem.is_some()
    }
    /// Whether device memory is initialized. Mirrors CLIc's `initialized()`.
    pub fn initialized(&self) -> bool {
        self.is_allocated()
    }
    /// Whether this array owns its device allocation. Mirrors CLIc's `ownsMemory()`.
    pub fn owns_memory(&self) -> bool {
        self.owns_memory
    }

    /// Return the raw GPU memory pointer — used by `execution.rs`.
    pub fn mem_ptr(&self) -> Option<&GpuMemPtr> {
        self.mem.as_ref()
    }
    /// Return the shared GPU memory handle. Mirrors CLIc's `get_ptr()`.
    pub fn get_ptr(&self) -> Option<GpuMemPtr> {
        self.mem.clone()
    }

    fn validate_region(&self, region: [usize; 3], origin: [usize; 3]) -> Result<()> {
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
        Ok(())
    }
}

fn copy_region_bytes(
    src: &[u8],
    src_shape: [usize; 3],
    dst: &mut [u8],
    dst_shape: [usize; 3],
    src_origin: [usize; 3],
    dst_origin: [usize; 3],
    region: [usize; 3],
    item_size: usize,
) {
    for z in 0..region[2] {
        for y in 0..region[1] {
            let src_index = linear_index(
                src_origin[0],
                src_origin[1] + y,
                src_origin[2] + z,
                src_shape,
            ) * item_size;
            let dst_index = linear_index(
                dst_origin[0],
                dst_origin[1] + y,
                dst_origin[2] + z,
                dst_shape,
            ) * item_size;
            let byte_count = region[0] * item_size;
            dst[dst_index..dst_index + byte_count]
                .copy_from_slice(&src[src_index..src_index + byte_count]);
        }
    }
}

fn linear_index(x: usize, y: usize, z: usize, shape: [usize; 3]) -> usize {
    x + y * shape[0] + z * shape[0] * shape[1]
}

// ── Convenience free functions ────────────────────────────────────────────────

/// Push a typed slice from the host to the GPU, returning an `ArrayPtr`.
pub fn push<T: GpuScalar>(
    data: &[T],
    width: usize,
    height: usize,
    depth: usize,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    let dim = shape_to_dimension(width, height, depth);
    Array::create_with_data(width, height, depth, dim, MType::Buffer, data, device)
}

/// Pull a typed array from the GPU back to a host `Vec`.
pub fn pull<T: GpuScalar>(arr: &ArrayPtr) -> Result<Vec<T>> {
    let lock = arr.lock().unwrap();
    let byte_count = lock.size() * std::mem::size_of::<T>();
    let mut bytes = vec![0u8; byte_count];
    lock.read_to_bytes(&mut bytes)?;
    // Safety: T: GpuScalar is Copy + 'static, and all bit patterns from GPU are valid
    // for the numeric types (f32, i32, u32, etc.) that implement GpuScalar.
    let mut out = Vec::with_capacity(lock.size());
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const T, out.as_mut_ptr(), lock.size());
        out.set_len(lock.size());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_region_bytes_copies_rectangular_rows() {
        let src: Vec<u8> = (0..16).collect();
        let mut dst = vec![100u8; 16];

        copy_region_bytes(
            &src,
            [4, 4, 1],
            &mut dst,
            [4, 4, 1],
            [1, 1, 0],
            [0, 2, 0],
            [2, 2, 1],
            1,
        );

        assert_eq!(&dst[8..10], &[5, 6]);
        assert_eq!(&dst[12..14], &[9, 10]);
        assert_eq!(dst[10], 100);
    }

    #[test]
    fn copy_region_bytes_respects_item_size() {
        let src: Vec<u8> = (0..24).collect();
        let mut dst = vec![0u8; 24];

        copy_region_bytes(
            &src,
            [3, 2, 1],
            &mut dst,
            [3, 2, 1],
            [1, 0, 0],
            [0, 1, 0],
            [2, 1, 1],
            4,
        );

        assert_eq!(&dst[12..20], &src[4..12]);
    }
}
