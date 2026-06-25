use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::DType;

const KERNEL_SOURCE: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void math_trigo(
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

fn apply_trigonometric_op(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    op_expr: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let global_range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let kernel = ("math_trigo", KERNEL_SOURCE);
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![("OP(x)", ConstantValue::Str(op_expr.to_string()))];
    execute(device, kernel, &params, global_range, [0, 0, 0], &constants)?;
    Ok(dst)
}

pub fn sin(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_trigonometric_op(device, src, dst, "sin(x)")
}

pub fn cos(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_trigonometric_op(device, src, dst, "cos(x)")
}

pub fn tan(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_trigonometric_op(device, src, dst, "tan(x)")
}

pub fn asin(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_trigonometric_op(device, src, dst, "asin(x)")
}

pub fn acos(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_trigonometric_op(device, src, dst, "acos(x)")
}

pub fn atan(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_trigonometric_op(device, src, dst, "atan(x)")
}

pub fn sinh(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_trigonometric_op(device, src, dst, "sinh(x)")
}

pub fn cosh(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_trigonometric_op(device, src, dst, "cosh(x)")
}

pub fn tanh(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_trigonometric_op(device, src, dst, "tanh(x)")
}
