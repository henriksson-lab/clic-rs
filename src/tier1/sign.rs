use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;

use super::common;

pub fn sign(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "(x > 0) ? 1 : ((x < 0) ? -1 : 0)")
}
