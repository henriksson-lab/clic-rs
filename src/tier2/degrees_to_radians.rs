use std::f32::consts::PI;

use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::types::DType;

/// Convert degrees to radians: src * pi / 180.
pub fn degrees_to_radians(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    tier1::multiply_image_and_scalar(device, src, Some(dst), PI / 180.0)
}
