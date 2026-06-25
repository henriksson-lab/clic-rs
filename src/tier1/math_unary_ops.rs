use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::DType;

const KERNEL_SOURCE: &str = r#"
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

fn apply_unary_math_operation(
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
    let kernel = ("math_unary", KERNEL_SOURCE);
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![("OP(x)", ConstantValue::Str(op_expr.to_string()))];
    execute(device, kernel, &params, global_range, [0, 0, 0], &constants)?;
    Ok(dst)
}

pub fn absolute(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "fabs(x)")
}

pub fn cubic_root(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "cbrt(x)")
}

pub fn square_root(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "sqrt(x)")
}

pub fn exponential(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "exp(x)")
}

pub fn exponential2(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "exp2(x)")
}

pub fn exponential10(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "exp10(x)")
}

pub fn logarithm(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "log(x)")
}

pub fn logarithm2(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "log2(x)")
}

pub fn logarithm10(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "log10(x)")
}

pub fn reciprocal(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "1.0f / x")
}

pub fn ceil(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "ceil(x)")
}

pub fn floor(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "floor(x)")
}

pub fn round(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "round(x)")
}

pub fn truncate(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "trunc(x)")
}

pub fn binary_not(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    apply_unary_math_operation(device, src, dst, "(x != 0) ? 0 : 1")
}
