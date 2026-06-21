use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

pub fn grayscale_opening(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let tmp = tier1::minimum_filter(device, src, None, rx, ry, rz, connectivity)?;
    tier1::maximum_filter(device, &tmp, dst, rx, ry, rz, connectivity)
}

/// Grayscale opening (min filter then max filter) with box connectivity.
pub fn opening_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
) -> Result<ArrayPtr> {
    grayscale_opening(device, src, dst, rx, ry, rz, "box")
}

/// Grayscale opening with sphere connectivity.
pub fn opening_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
) -> Result<ArrayPtr> {
    grayscale_opening(device, src, dst, rx, ry, rz, "sphere")
}
