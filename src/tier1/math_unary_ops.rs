use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;

use super::common;

pub fn absolute(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "fabs(x)")
}

pub fn cubic_root(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "cbrt(x)")
}

pub fn square_root(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "sqrt(x)")
}

pub fn exponential(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "exp(x)")
}

pub fn exponential2(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "exp2(x)")
}

pub fn exponential10(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "exp10(x)")
}

pub fn logarithm(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "log(x)")
}

pub fn logarithm2(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "log2(x)")
}

pub fn logarithm10(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "log10(x)")
}

pub fn reciprocal(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "1.0f / x")
}

pub fn ceil(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "ceil(x)")
}

pub fn floor(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "floor(x)")
}

pub fn round(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "round(x)")
}

pub fn truncate(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "trunc(x)")
}

pub fn binary_not(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    common::unary_math_op(device, src, dst, "(x != 0) ? 0 : 1")
}
