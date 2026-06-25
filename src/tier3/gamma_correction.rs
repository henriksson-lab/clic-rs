use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{evaluate, ParameterValue};
use crate::tier0;
use crate::tier2;
use crate::types::DType;

/// Gamma correction: `(src / max)^gamma * max`.
pub fn gamma_correction(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    gamma: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let max_intensity = tier2::maximum_of_all_pixels(device, src)?;
    evaluate(
        device,
        "pow(a / m, g) * m",
        &[
            ParameterValue::Array(src.clone()),
            ParameterValue::Float(max_intensity),
            ParameterValue::Float(gamma),
        ],
        &dst,
    )?;
    Ok(dst)
}
