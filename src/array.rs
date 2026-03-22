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
    pub(crate) _owns_memory: bool,
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
            _owns_memory: true,
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
        let mem = BackendManager::get()
            .backend()
            .allocate_memory(&self.device, [self.width, self.height, self.depth], self.dtype, self.mtype)?;
        self.mem = Some(mem);
        Ok(())
    }

    // ── Data transfer ─────────────────────────────────────────────────────────

    pub fn write_from_typed<T: GpuScalar>(&self, data: &[T]) -> Result<()> {
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        let byte_size = data.len() * std::mem::size_of::<T>();
        BackendManager::get().backend().write_memory(
            &self.device,
            mem,
            [self.width, self.height, self.depth],
            [0, 0, 0],
            data.as_ptr() as *const u8,
            byte_size,
        )
    }

    pub fn write_from_bytes(&self, data: &[u8]) -> Result<()> {
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        BackendManager::get().backend().write_memory(
            &self.device,
            mem,
            [self.width, self.height, self.depth],
            [0, 0, 0],
            data.as_ptr(),
            data.len(),
        )
    }

    pub fn read_to_typed<T: GpuScalar>(&self, data: &mut [T]) -> Result<()> {
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        let byte_size = data.len() * std::mem::size_of::<T>();
        BackendManager::get().backend().read_memory(
            &self.device,
            mem,
            [self.width, self.height, self.depth],
            [0, 0, 0],
            data.as_mut_ptr() as *mut u8,
            byte_size,
        )
    }

    pub fn read_to_bytes(&self, data: &mut [u8]) -> Result<()> {
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        BackendManager::get().backend().read_memory(
            &self.device,
            mem,
            [self.width, self.height, self.depth],
            [0, 0, 0],
            data.as_mut_ptr(),
            data.len(),
        )
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

    pub fn fill(&self, value: f32) -> Result<()> {
        let mem = self.mem.as_ref().ok_or(CleError::NotAllocated)?;
        BackendManager::get()
            .backend()
            .set_memory(&self.device, mem, value, self.dtype, self.size())
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn depth(&self) -> usize { self.depth }
    pub fn dim(&self) -> usize { self.dim }
    /// Effective dimensionality derived from shape (may differ from `dim()`).
    pub fn dimension(&self) -> usize { shape_to_dimension(self.width, self.height, self.depth) }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn mtype(&self) -> MType { self.mtype }
    pub fn device(&self) -> &DeviceArc { &self.device }
    pub fn size(&self) -> usize { self.width * self.height * self.depth }
    pub fn byte_size(&self) -> usize { self.size() * self.dtype.byte_size() }
    pub fn is_allocated(&self) -> bool { self.mem.is_some() }

    /// Return the raw GPU memory pointer — used by `execution.rs`.
    pub fn mem_ptr(&self) -> Option<&GpuMemPtr> { self.mem.as_ref() }
}

// ── Convenience free functions ────────────────────────────────────────────────

/// Push a typed slice from the host to the GPU, returning an `ArrayPtr`.
pub fn push<T: GpuScalar>(data: &[T], width: usize, height: usize, depth: usize, device: &DeviceArc) -> Result<ArrayPtr> {
    let dim = shape_to_dimension(width, height, depth);
    Array::create_with_data(width, height, depth, dim, MType::Buffer, data, device)
}

/// Pull a typed array from the GPU back to a host `Vec`.
pub fn pull<T: GpuScalar>(arr: &ArrayPtr) -> Result<Vec<T>> {
    let lock = arr.lock().unwrap();
    let mut out = vec![unsafe { std::mem::zeroed::<T>() }; lock.size()];
    lock.read_to_typed(&mut out)?;
    Ok(out)
}
