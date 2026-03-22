//! Tier 3 — operations composing tier1 and tier2 primitives.
//!
//! Mirrors CLIc's `clic/src/tier3/` directory.

use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;
use crate::tier2;

/// Return the mean pixel value of the entire array.
pub fn mean_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let sum = tier2::sum_of_all_pixels(device, src)?;
    let n = src.lock().unwrap().size();
    Ok(sum / n as f32)
}

/// Gamma correction: `(src / max)^gamma * max`.
pub fn gamma_correction(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, gamma: f32) -> Result<ArrayPtr> {
    let max = tier2::maximum_of_all_pixels(device, src)?;
    let norm = tier2::divide_image_by_scalar(device, src, None, max)?;
    let powered = tier1::power(device, &norm, None, gamma)?;
    tier1::multiply_image_and_scalar(device, &powered, dst, max)
}
