/// Tier 0 — array creation helpers (mirrors CLIc's `tier0.cpp`).
use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::types::{DType, MType};
use crate::utils::shape_to_dimension;

/// Ensure `dst` is allocated like `src` (same shape, same dtype, or `force_dtype`).
/// If `dst` is `None`, creates a new array.
pub fn create_like(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    force_dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    let dtype = if force_dtype == DType::Unknown {
        s.dtype()
    } else {
        force_dtype
    };
    let width = s.width();
    let height = s.height();
    let depth = s.depth();
    drop(s);
    create_dst(src, dst, width, height, depth, dtype, device)
}

/// Create a destination array with explicit shape, preserving source memory type.
pub fn create_dst(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    width: usize,
    height: usize,
    depth: usize,
    force_dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    if let Some(d) = dst {
        return Ok(d);
    }
    let s = src.lock().unwrap();
    let dtype = if force_dtype == DType::Unknown {
        s.dtype()
    } else {
        force_dtype
    };
    let dim = shape_to_dimension(width, height, depth);
    Array::create(width, height, depth, dim, dtype, s.mtype(), device)
}

/// Convenience: create like src preserving dtype.
pub(crate) fn create_like_same(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    create_like(src, dst, DType::Unknown, device)
}

/// Create a 1×1×1 float array (used for scalar outputs).
pub fn create_one(device: &DeviceArc) -> Result<ArrayPtr> {
    Array::create(1, 1, 1, 1, DType::Float, MType::Buffer, device)
}

/// Create a 1x1x1 array based on a source array.
pub fn create_one_like(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    force_dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    create_dst(src, dst, 1, 1, 1, force_dtype, device)
}

/// Create a 1D vector array with `length` elements.
pub fn create_vector(length: usize, dtype: DType, device: &DeviceArc) -> Result<ArrayPtr> {
    Array::create(length, 1, 1, 1, dtype, MType::Buffer, device)
}

/// Create a 1D vector array based on a source array.
pub fn create_vector_like(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    length: usize,
    force_dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    create_dst(src, dst, length, 1, 1, force_dtype, device)
}

/// Create a destination with shape (x, y, 1) from `src`.
pub fn create_xy(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    force_dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    let width = s.width();
    let height = s.height();
    drop(s);
    create_dst(src, dst, width, height, 1, force_dtype, device)
}

/// Create a destination with shape (y, x, 1) from `src`.
pub fn create_yx(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    force_dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    let height = s.height();
    let width = s.width();
    drop(s);
    create_dst(src, dst, height, width, 1, force_dtype, device)
}

/// Create a destination with shape (z, y, 1) from `src`.
pub fn create_zy(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    force_dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    let depth = s.depth();
    let height = s.height();
    drop(s);
    create_dst(src, dst, depth, height, 1, force_dtype, device)
}

/// Create a destination with shape (y, z, 1) from `src`.
pub fn create_yz(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    force_dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    let height = s.height();
    let depth = s.depth();
    drop(s);
    create_dst(src, dst, height, depth, 1, force_dtype, device)
}

/// Create a destination with shape (x, z, 1) from `src`.
pub fn create_xz(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    force_dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    let width = s.width();
    let depth = s.depth();
    drop(s);
    create_dst(src, dst, width, depth, 1, force_dtype, device)
}

/// Create a destination with shape (z, x, 1) from `src`.
pub fn create_zx(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    force_dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    let s = src.lock().unwrap();
    let depth = s.depth();
    let width = s.width();
    drop(s);
    create_dst(src, dst, depth, width, 1, force_dtype, device)
}
