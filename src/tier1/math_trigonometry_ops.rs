use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;

use super::common;

pub fn sin(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "sin(x)")
}

pub fn cos(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "cos(x)")
}

pub fn tan(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "tan(x)")
}

pub fn asin(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "asin(x)")
}

pub fn acos(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "acos(x)")
}

pub fn atan(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "atan(x)")
}

pub fn sinh(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "sinh(x)")
}

pub fn cosh(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "cosh(x)")
}

pub fn tanh(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "tanh(x)")
}
