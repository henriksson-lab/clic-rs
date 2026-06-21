use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier2;
use crate::tier3;

/// Mean Squared Error between two arrays: mean((src0 - src1)^2).
pub fn mean_squared_error(device: &DeviceArc, src0: &ArrayPtr, src1: &ArrayPtr) -> Result<f32> {
    let diff_sq = tier2::squared_difference(device, src0, src1, None)?;
    tier3::mean_of_all_pixels(device, &diff_sq)
}
