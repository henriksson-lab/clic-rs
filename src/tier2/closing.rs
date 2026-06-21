use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

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
