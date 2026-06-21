use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::tier4;
use crate::types::DType;

/// Normalize image values to `[0, 1]`.
///
/// Mirrors CLIc's `normalize_func`.
pub fn normalize(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    low_percentile: f32,
    high_percentile: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let min = if low_percentile > 0.0 {
        tier4::percentile(device, src, low_percentile)?.round()
    } else {
        tier2::minimum_of_all_pixels(device, src)?
    };
    let max = if high_percentile > 0.0 {
        tier4::percentile(device, src, high_percentile)?.round()
    } else {
        tier2::maximum_of_all_pixels(device, src)?
    };

    let scale = 1.0 / (max - min);
    let shifted = tier1::subtract_scalar_from_image(device, src, None, min)?;
    let scaled = tier1::multiply_image_and_scalar(device, &shifted, None, scale)?;
    let lower_clamped = tier1::maximum_image_and_scalar(device, &scaled, None, 0.0)?;
    tier1::minimum_image_and_scalar(device, &lower_clamped, Some(dst), 1.0)
}
