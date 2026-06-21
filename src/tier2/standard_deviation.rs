use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

pub fn standard_deviation(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let var = tier1::variance_filter(device, src, None, rx, ry, rz, connectivity)?;
    tier1::power(device, &var, dst, 0.5)
}

/// Standard deviation filter with box connectivity.
pub fn standard_deviation_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
) -> Result<ArrayPtr> {
    standard_deviation(device, src, dst, rx, ry, rz, "box")
}

/// Standard deviation filter with sphere connectivity.
pub fn standard_deviation_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
) -> Result<ArrayPtr> {
    standard_deviation(device, src, dst, rx, ry, rz, "sphere")
}
