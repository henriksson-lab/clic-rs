use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

/// Negate: -src.
pub fn invert(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    tier1::multiply_image_and_scalar(device, src, dst, -1.0)
}
