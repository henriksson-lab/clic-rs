use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;

const MATH_UNARY_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void math_unary(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  float res = OP(value);
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(res));
}
"#;

const MATH_BINARY_SCALAR_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void math_binary_scalar(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    float scalar
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  const float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  float res = OP(value, scalar);
  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(res));
}
"#;

pub const IMAGE_OPERATION_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#ifndef APPLY_OP
  #error "APPLY_OP must be defined (e.g. #define APPLY_OP(x,y) (x+y))"
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

pub(crate) fn unary_math_op(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    op_expr: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = {
        let lock = dst.lock().unwrap();
        [lock.width(), lock.height(), lock.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![("OP(x)", ConstantValue::Str(op_expr.to_string()))];
    execute(
        device,
        ("math_unary", MATH_UNARY_SRC),
        &params,
        global,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub(crate) fn binary_scalar_op(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
    op_expr: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar", ParameterValue::Float(scalar)),
    ];
    let constants = vec![("OP(x,y)", ConstantValue::Str(op_expr.to_string()))];
    execute(
        device,
        ("math_binary_scalar", MATH_BINARY_SCALAR_SRC),
        &params,
        global,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub(crate) fn image_op(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
    op_expr: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src0, dst, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![("APPLY_OP(x,y)", ConstantValue::Str(op_expr.to_string()))];
    execute(
        device,
        ("image_operation", IMAGE_OPERATION_SRC),
        &params,
        global,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}
