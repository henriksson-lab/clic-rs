use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;
use crate::tier2;

/// Gamma correction: `(src / max)^gamma * max`.
pub fn gamma_correction(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    gamma: f32,
) -> Result<ArrayPtr> {
    let max = tier2::maximum_of_all_pixels(device, src)?;
    let norm = tier1::divide_image_by_scalar(device, src, None, max)?;
    let powered = tier1::power(device, &norm, None, gamma)?;
    tier1::multiply_image_and_scalar(device, &powered, dst, max)
}
