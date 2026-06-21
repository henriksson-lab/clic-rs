use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

pub fn divide_by_gaussian_background(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    sigma_x: f32,
    sigma_y: f32,
    sigma_z: f32,
) -> Result<ArrayPtr> {
    let temp = tier1::gaussian_blur(device, src, None, sigma_x, sigma_y, sigma_z)?;
    tier1::divide_images(device, src, &temp, dst)
}
