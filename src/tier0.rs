/// Tier 0 — array creation helpers (mirrors CLIc's `tier0.cpp`).
use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::types::{DType, MType};
use crate::utils::shape_to_dimension;

/// Check whether `dst` already exists and resolve an unknown dtype from `src`.
/// Mirrors CLIc's `check_and_set()` helper.
pub fn check_and_set(src: &ArrayPtr, dst: &Option<ArrayPtr>, dtype: &mut DType) -> bool {
    if dst.is_some() {
        return true;
    }
    if *dtype == DType::Unknown {
        *dtype = src.lock().unwrap().dtype;
    }
    false
}

/// Create a destination array with explicit shape, preserving source memory type.
pub fn create_dst(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    width: usize,
    height: usize,
    depth: usize,
    mut force_dtype: DType,
    _device: &DeviceArc,
) -> Result<ArrayPtr> {
    if check_and_set(src, &dst, &mut force_dtype) {
        return Ok(dst.unwrap());
    }
    let s = src.lock().unwrap();
    let mtype = s.mtype;
    let device = s.device.clone();
    drop(s);
    let dim = shape_to_dimension(width, height, depth);
    Array::create(width, height, depth, dim, force_dtype, mtype, &device)
}

/// Ensure `dst` is allocated like `src` (same shape, same dtype, or `force_dtype`).
/// If `dst` is `None`, creates a new array.
pub fn create_like(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mut force_dtype: DType,
    _device: &DeviceArc,
) -> Result<ArrayPtr> {
    if check_and_set(src, &dst, &mut force_dtype) {
        return Ok(dst.unwrap());
    }
    let s = src.lock().unwrap();
    let width = s.width;
    let height = s.height;
    let depth = s.depth;
    let dim = s.dim;
    let mtype = s.mtype;
    let device = s.device.clone();
    drop(s);
    Array::create(width, height, depth, dim, force_dtype, mtype, &device)
}

/// Create a 1x1x1 array based on a source array.
pub fn create_one(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mut force_dtype: DType,
    _device: &DeviceArc,
) -> Result<ArrayPtr> {
    if check_and_set(src, &dst, &mut force_dtype) {
        return Ok(dst.unwrap());
    }
    let device = src.lock().unwrap().device.clone();
    Array::create(1, 1, 1, 1, force_dtype, MType::Buffer, &device)
}

/// Create a 1D vector array based on a source array.
pub fn create_vector(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    length: usize,
    mut force_dtype: DType,
    _device: &DeviceArc,
) -> Result<ArrayPtr> {
    if check_and_set(src, &dst, &mut force_dtype) {
        return Ok(dst.unwrap());
    }
    let device = src.lock().unwrap().device.clone();
    Array::create(length, 1, 1, 1, force_dtype, MType::Buffer, &device)
}

/// Create a destination with shape (x, y, 1) from `src`.
pub fn create_xy(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mut force_dtype: DType,
    _device: &DeviceArc,
) -> Result<ArrayPtr> {
    if check_and_set(src, &dst, &mut force_dtype) {
        return Ok(dst.unwrap());
    }
    let s = src.lock().unwrap();
    let width = s.width;
    let height = s.height;
    let mtype = s.mtype;
    let device = s.device.clone();
    drop(s);
    let dim = shape_to_dimension(width, height, 1);
    Array::create(width, height, 1, dim, force_dtype, mtype, &device)
}

/// Create a destination with shape (y, x, 1) from `src`.
pub fn create_yx(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mut force_dtype: DType,
    _device: &DeviceArc,
) -> Result<ArrayPtr> {
    if check_and_set(src, &dst, &mut force_dtype) {
        return Ok(dst.unwrap());
    }
    let s = src.lock().unwrap();
    let height = s.height;
    let width = s.width;
    let mtype = s.mtype;
    let device = s.device.clone();
    drop(s);
    let dim = shape_to_dimension(height, width, 1);
    Array::create(height, width, 1, dim, force_dtype, mtype, &device)
}

/// Create a destination with shape (z, y, 1) from `src`.
pub fn create_zy(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mut force_dtype: DType,
    _device: &DeviceArc,
) -> Result<ArrayPtr> {
    if check_and_set(src, &dst, &mut force_dtype) {
        return Ok(dst.unwrap());
    }
    let s = src.lock().unwrap();
    let depth = s.depth;
    let height = s.height;
    let mtype = s.mtype;
    let device = s.device.clone();
    drop(s);
    let dim = shape_to_dimension(depth, height, 1);
    Array::create(depth, height, 1, dim, force_dtype, mtype, &device)
}

/// Create a destination with shape (y, z, 1) from `src`.
pub fn create_yz(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mut force_dtype: DType,
    _device: &DeviceArc,
) -> Result<ArrayPtr> {
    if check_and_set(src, &dst, &mut force_dtype) {
        return Ok(dst.unwrap());
    }
    let s = src.lock().unwrap();
    let height = s.height;
    let depth = s.depth;
    let mtype = s.mtype;
    let device = s.device.clone();
    drop(s);
    let dim = shape_to_dimension(height, depth, 1);
    Array::create(height, depth, 1, dim, force_dtype, mtype, &device)
}

/// Create a destination with shape (x, z, 1) from `src`.
pub fn create_xz(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mut force_dtype: DType,
    _device: &DeviceArc,
) -> Result<ArrayPtr> {
    if check_and_set(src, &dst, &mut force_dtype) {
        return Ok(dst.unwrap());
    }
    let s = src.lock().unwrap();
    let width = s.width;
    let depth = s.depth;
    let mtype = s.mtype;
    let device = s.device.clone();
    drop(s);
    let dim = shape_to_dimension(width, depth, 1);
    Array::create(width, depth, 1, dim, force_dtype, mtype, &device)
}

/// Create a destination with shape (z, x, 1) from `src`.
pub fn create_zx(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mut force_dtype: DType,
    _device: &DeviceArc,
) -> Result<ArrayPtr> {
    if check_and_set(src, &dst, &mut force_dtype) {
        return Ok(dst.unwrap());
    }
    let s = src.lock().unwrap();
    let depth = s.depth;
    let width = s.width;
    let mtype = s.mtype;
    let device = s.device.clone();
    drop(s);
    let dim = shape_to_dimension(depth, width, 1);
    Array::create(depth, width, 1, dim, force_dtype, mtype, &device)
}
