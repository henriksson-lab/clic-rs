use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

pub fn standard_deviation(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let temp = tier1::variance_filter(
        device,
        src,
        None,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )?;
    tier1::power(device, &temp, dst, 0.5)
}

/// Standard deviation filter with sphere connectivity.
pub fn standard_deviation_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    standard_deviation(device, src, dst, radius_x, radius_y, radius_z, "sphere")
}

/// Standard deviation filter with box connectivity.
pub fn standard_deviation_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    standard_deviation(device, src, dst, radius_x, radius_y, radius_z, "box")
}
