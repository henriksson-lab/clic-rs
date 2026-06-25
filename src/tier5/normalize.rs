use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{evaluate, ParameterValue};
use crate::tier0;
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
    let new_min = 0.0f32;
    let new_max = 1.0f32;
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
    let constant = -(new_max - new_min) / (max - min);
    evaluate(
        device,
        "fmin(fmax((lo - a) * c, 0.0f), 1.0f)",
        &[
            ParameterValue::Float(min),
            ParameterValue::Array(src.clone()),
            ParameterValue::Float(constant),
        ],
        &dst,
    )?;
    Ok(dst)
}
