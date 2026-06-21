use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;

use super::common;

pub use super::common::IMAGE_OPERATION_SRC;

pub fn maximum_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "fmax(x, y)")
}

pub fn minimum_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "fmin(x, y)")
}

pub fn multiply_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "x * y")
}

pub fn divide_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "x / y")
}

pub fn modulo_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "fmod(x, y)")
}

pub fn power_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "pow(x, y)")
}

pub fn greater(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "(x > y) ? 1 : 0")
}

pub fn greater_or_equal(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "(x >= y) ? 1 : 0")
}

pub fn smaller(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "(x < y) ? 1 : 0")
}

pub fn smaller_or_equal(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "(x <= y) ? 1 : 0")
}

pub fn equal(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "(x == y) ? 1 : 0")
}

pub fn not_equal(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "(x != y) ? 1 : 0")
}

pub fn binary_and(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "(x != 0 && y != 0) ? 1 : 0")
}

pub fn binary_or(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "(x != 0 || y != 0) ? 1 : 0")
}

pub fn binary_xor(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "((x != 0) != (y != 0)) ? 1 : 0")
}

pub fn binary_subtract(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    common::image_op(device, src0, src1, dst, "(x != 0 && y == 0) ? 1 : 0")
}
