/// Tier 0 — array creation helpers (mirrors CLIc's `tier0.cpp`).
use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::types::{DType, MType};
use crate::utils::shape_to_dimension;

/// Ensure `dst` is allocated like `src` (same shape, same dtype, or `force_dtype`).
/// If `dst` is `None`, creates a new array.
pub fn create_like(src: &ArrayPtr, dst: Option<ArrayPtr>, force_dtype: DType, device: &DeviceArc) -> Result<ArrayPtr> {
    if let Some(d) = dst {
        return Ok(d);
    }
    let s = src.lock().unwrap();
    let dtype = if force_dtype == DType::Unknown { s.dtype() } else { force_dtype };
    drop(s);
    Array::create_like_typed(src, dtype, device)
}

/// Convenience: create like src preserving dtype.
pub fn create_like_same(src: &ArrayPtr, dst: Option<ArrayPtr>, device: &DeviceArc) -> Result<ArrayPtr> {
    create_like(src, dst, DType::Unknown, device)
}

/// Create a 1×1×1 float array (used for scalar outputs).
pub fn create_one(device: &DeviceArc) -> Result<ArrayPtr> {
    Array::create(1, 1, 1, 1, DType::Float, MType::Buffer, device)
}

/// Create a 1D vector array with `length` elements.
pub fn create_vector(length: usize, dtype: DType, device: &DeviceArc) -> Result<ArrayPtr> {
    Array::create(length, 1, 1, 1, dtype, MType::Buffer, device)
}

/// Create a 1D array of `n` float zeros.
pub fn create_float_vector(n: usize, device: &DeviceArc) -> Result<ArrayPtr> {
    create_vector(n, DType::Float, device)
}

/// Create an array with swapped X and Y axes relative to `src`.
pub fn create_xy_transposed(src: &ArrayPtr, device: &DeviceArc) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    Array::create(s.height(), s.width(), s.depth(), s.dim(), s.dtype(), s.mtype(), device)
}

/// Create an array with swapped X and Z axes relative to `src`.
pub fn create_xz_transposed(src: &ArrayPtr, device: &DeviceArc) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    Array::create(s.depth(), s.height(), s.width(), s.dim(), s.dtype(), s.mtype(), device)
}

/// Create an array with swapped Y and Z axes relative to `src`.
pub fn create_yz_transposed(src: &ArrayPtr, device: &DeviceArc) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    Array::create(s.width(), s.depth(), s.height(), s.dim(), s.dtype(), s.mtype(), device)
}

/// Create a destination array matching the projected shape along `axis`.
/// E.g. for a maximum-projection along Z (axis=2) on a 10×20×5 array → 10×20×1.
pub fn create_projection(src: &ArrayPtr, axis: usize, device: &DeviceArc) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    let (w, h, d) = match axis {
        0 => (1, s.height(), s.depth()),
        1 => (s.width(), 1, s.depth()),
        2 => (s.width(), s.height(), 1),
        _ => return Err(crate::error::CleError::Other("Invalid axis".into())),
    };
    let dim = shape_to_dimension(w, h, d);
    Array::create(w, h, d, dim, s.dtype(), s.mtype(), device)
}
