use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

pub fn bottom_hat(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let temp1 = tier1::maximum_filter(
        device,
        src,
        None,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )?;
    let temp2 = tier1::minimum_filter(
        device,
        &temp1,
        None,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )?;
    tier1::add_images_weighted(device, &temp2, src, dst, 1.0, -1.0)
}

/// Bottom-hat (black-hat): closing - src.
pub fn bottom_hat_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    bottom_hat(device, src, dst, radius_x, radius_y, radius_z, "box")
}

/// Bottom-hat with sphere connectivity.
pub fn bottom_hat_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    bottom_hat(device, src, dst, radius_x, radius_y, radius_z, "sphere")
}
