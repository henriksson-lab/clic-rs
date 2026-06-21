use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

pub fn subtract_gaussian_background(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    sigma_x: f32,
    sigma_y: f32,
    sigma_z: f32,
) -> Result<ArrayPtr> {
    let temp = tier1::gaussian_blur(device, src, None, sigma_x, sigma_y, sigma_z)?;
    tier1::add_images_weighted(device, src, &temp, dst, 1.0, -1.0)
}
