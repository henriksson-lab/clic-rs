use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{evaluate, ParameterValue};
use crate::tier0;
use crate::types::DType;

use super::minmax_of_all_pixels::{maximum_of_all_pixels, minimum_of_all_pixels};

/// Clamp pixel values to [min_intensity, max_intensity].
pub fn clip(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    min_intensity: f32,
    max_intensity: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let min_intensity = if min_intensity.is_nan() {
        minimum_of_all_pixels(device, src)?
    } else {
        min_intensity
    };
    let max_intensity = if max_intensity.is_nan() {
        maximum_of_all_pixels(device, src)?
    } else {
        max_intensity
    };
    evaluate(
        device,
        "clamp(a, lo, hi)",
        &[
            ParameterValue::Array(src.clone()),
            ParameterValue::Float(min_intensity),
            ParameterValue::Float(max_intensity),
        ],
        &dst,
    )?;
    Ok(dst)
}
