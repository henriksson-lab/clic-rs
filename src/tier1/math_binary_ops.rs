use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;

use super::common;

pub fn add_image_and_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "x + y")
}

pub fn subtract_scalar_from_image(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "x - y")
}

pub fn subtract_image_from_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "y - x")
}

pub fn multiply_image_and_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "x * y")
}

pub fn divide_image_by_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "x / y")
}

pub fn divide_scalar_by_image(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "y / x")
}

pub fn power(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "pow(x, y)")
}

pub fn root(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "rootn(x, y)")
}

pub fn maximum_image_and_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "fmax(x, y)")
}

pub fn minimum_image_and_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "fmin(x, y)")
}

pub fn greater_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "(x > y) ? 1 : 0")
}

pub fn greater_or_equal_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "(x >= y) ? 1 : 0")
}

pub fn smaller_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "(x < y) ? 1 : 0")
}

pub fn smaller_or_equal_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "(x <= y) ? 1 : 0")
}

pub fn equal_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "(x == y) ? 1 : 0")
}

pub fn not_equal_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    common::binary_scalar_op(device, src, dst, scalar, "(x != y) ? 1 : 0")
}
