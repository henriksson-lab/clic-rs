use std::f32::consts::PI;

use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::types::DType;

/// Convert radians to degrees: src * 180 / pi.
pub fn radians_to_degrees(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    tier1::multiply_image_and_scalar(device, src, Some(dst), 180.0 / PI)
}
