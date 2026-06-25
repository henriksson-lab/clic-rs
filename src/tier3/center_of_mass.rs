use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::{tier1, tier2};

/// Return the intensity-weighted center of mass as `[x, y, z]`.
///
/// Mirrors CLIc's `center_of_mass_func`, computing
/// `sum(position * intensity) / sum(intensity)` for each axis.
pub fn center_of_mass(device: &DeviceArc, src: &ArrayPtr) -> Result<Vec<f32>> {
    let sum = tier2::sum_of_all_pixels(device, src)?;
    let mut temp = tier1::multiply_image_and_position(device, src, None, 0)?;
    let sum_x = tier2::sum_of_all_pixels(device, &temp)?;
    temp = tier1::multiply_image_and_position(device, src, None, 1)?;
    let sum_y = tier2::sum_of_all_pixels(device, &temp)?;
    temp = tier1::multiply_image_and_position(device, src, None, 2)?;
    let sum_z = tier2::sum_of_all_pixels(device, &temp)?;
    Ok(vec![sum_x / sum, sum_y / sum, sum_z / sum])
}
