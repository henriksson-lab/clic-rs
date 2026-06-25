use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::DType;

pub const IMAGE_OPERATION_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#ifndef APPLY_OP
  #error "APPLY_OP must be defined as a macro (e.g., #define APPLY_OP(x,y) (x+y))"
#endif

__kernel void image_operation(
    IMAGE_src0_TYPE src0,
    IMAGE_src1_TYPE src1,
    IMAGE_dst_TYPE  dst
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    const float value0 = (float) READ_IMAGE(src0, sampler, POS_src0_INSTANCE(x,y,z,0)).x;
    const float value1 = (float) READ_IMAGE(src1, sampler, POS_src1_INSTANCE(x,y,z,0)).x;
    const float res = APPLY_OP(value0, value1);

    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(res));
}
"#;

fn apply_images_math_operation(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
    op_define: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src0, dst, DType::Unknown, device)?;
    let kernel = ("image_operation", IMAGE_OPERATION_SRC);
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let src0 = src0.lock().unwrap();
        [src0.width(), src0.height(), src0.depth()]
    };
    let constants = vec![("APPLY_OP(x,y)", ConstantValue::Str(op_define.to_string()))];
    execute(device, kernel, &params, range, [0, 0, 0], &constants)?;
    Ok(dst)
}

pub fn maximum_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "fmax(x, y)")
}

pub fn minimum_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "fmin(x, y)")
}

pub fn multiply_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x * y)")
}

pub fn divide_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x / y)")
}

pub fn modulo_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "fmod(x, y)")
}

pub fn power_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "pow(x, y)")
}

pub fn greater(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x > y) ? 1 : 0")
}

pub fn greater_or_equal(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x >= y) ? 1 : 0")
}

pub fn smaller(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x < y) ? 1 : 0")
}

pub fn smaller_or_equal(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x <= y) ? 1 : 0")
}

pub fn equal(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x == y) ? 1 : 0")
}

pub fn not_equal(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x != y) ? 1 : 0")
}

pub fn binary_and(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x != 0 && y != 0) ? 1 : 0")
}

pub fn binary_or(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x != 0 || y != 0) ? 1 : 0")
}

pub fn binary_xor(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "((x != 0) != (y != 0)) ? 1 : 0")
}

pub fn binary_subtract(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_images_math_operation(device, src0, src1, dst, "(x != 0 && y == 0) ? 1 : 0")
}
