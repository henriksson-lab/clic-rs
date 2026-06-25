use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

pub fn top_hat(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let temp1 = tier1::minimum_filter(
        device,
        src,
        None,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )?;
    let temp2 = tier1::maximum_filter(
        device,
        &temp1,
        None,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )?;
    tier1::add_images_weighted(device, src, &temp2, dst, 1.0, -1.0)
}

/// Top-hat: src - opening.
pub fn top_hat_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    top_hat(device, src, dst, radius_x, radius_y, radius_z, "box")
}

/// Top-hat with sphere connectivity.
pub fn top_hat_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    top_hat(device, src, dst, radius_x, radius_y, radius_z, "sphere")
}
