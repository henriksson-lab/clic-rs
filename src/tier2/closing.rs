use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier1;

pub fn closing(
    device: &DeviceArc,
    src: &ArrayPtr,
    footprint: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    if src.lock().unwrap().dimension() != footprint.lock().unwrap().dimension() {
        return Err(CleError::DimensionMismatch);
    }
    let tmp = tier1::dilation(device, src, footprint, None)?;
    tier1::erosion(device, &tmp, footprint, dst)
}

pub fn binary_closing(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let tmp = tier1::binary_dilate(device, src, None, rx, ry, rz, connectivity)?;
    tier1::binary_erode(device, &tmp, dst, rx, ry, rz, connectivity)
}

pub fn grayscale_closing(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let tmp = tier1::maximum_filter(device, src, None, rx, ry, rz, connectivity)?;
    tier1::minimum_filter(device, &tmp, dst, rx, ry, rz, connectivity)
}

/// Grayscale closing (max filter then min filter) with box connectivity.
pub fn closing_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
) -> Result<ArrayPtr> {
    grayscale_closing(device, src, dst, rx, ry, rz, "box")
}

/// Grayscale closing with sphere connectivity.
pub fn closing_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
) -> Result<ArrayPtr> {
    grayscale_closing(device, src, dst, rx, ry, rz, "sphere")
}
