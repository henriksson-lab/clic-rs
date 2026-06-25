use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::types::DType;

/// Gaussian(sigma1) - Gaussian(sigma2).
pub fn difference_of_gaussian(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    sigma1: [f32; 3],
    sigma2: [f32; 3],
) -> Result<ArrayPtr> {
    let [sigma1_x, sigma1_y, sigma1_z] = sigma1;
    let [sigma2_x, sigma2_y, sigma2_z] = sigma2;
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let gauss1 = tier1::gaussian_blur(device, src, None, sigma1_x, sigma1_y, sigma1_z)?;
    let gauss2 = tier1::gaussian_blur(device, src, None, sigma2_x, sigma2_y, sigma2_z)?;
    tier1::add_images_weighted(device, &gauss1, &gauss2, Some(dst), 1.0, -1.0)
}
