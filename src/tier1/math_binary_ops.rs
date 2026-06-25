use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::DType;

const KERNEL_SOURCE: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE |
                              CLK_FILTER_NEAREST;
#if DST_IS_INT
#define FINALIZE(x) rint(x)
#else
#define FINALIZE(x) (x)
#endif

__kernel void cle_binary_operation(
    IMAGE_src_TYPE src,
    IMAGE_dst_TYPE dst,
    const float    scalar
)
{
  int x = get_global_id(0);
  int y = get_global_id(1);
  int z = get_global_id(2);

  const float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
  const float res = APPLY_OP(value, scalar);

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(FINALIZE(res)));
}
"#;

fn apply_binary_math_operation(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
    op_define: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let dst_is_int = {
        let dst = dst.lock().unwrap();
        dst.dtype() != DType::Float
    };
    let kernel = ("cle_binary_operation", KERNEL_SOURCE);
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar", ParameterValue::Float(scalar)),
    ];
    let range = {
        let src = src.lock().unwrap();
        [src.width(), src.height(), src.depth()]
    };
    let constants = vec![
        ("APPLY_OP(x, y)", ConstantValue::Str(op_define.to_string())),
        ("DST_IS_INT", ConstantValue::Int(i32::from(dst_is_int))),
    ];

    execute(device, kernel, &params, range, [0, 0, 0], &constants)?;
    Ok(dst)
}

pub fn power(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "pow(x, y)")
}

pub fn root(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "rootn(x, y)")
}

pub fn maximum_image_and_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "fmax(x, y)")
}

pub fn minimum_image_and_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "fmin(x, y)")
}

pub fn add_image_and_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(x + y)")
}

pub fn subtract_scalar_from_image(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(x - y)")
}

pub fn subtract_image_from_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(y - x)")
}

pub fn divide_image_by_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(x / y)")
}

pub fn divide_scalar_by_image(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(y / x)")
}

pub fn multiply_image_and_scalar(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(x * y)")
}

pub fn greater_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(x > y ? 1.0f : 0.0f)")
}

pub fn greater_or_equal_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(x >= y ? 1.0f : 0.0f)")
}

pub fn smaller_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(x < y ? 1.0f : 0.0f)")
}

pub fn smaller_or_equal_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(x <= y ? 1.0f : 0.0f)")
}

pub fn equal_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(x == y ? 1.0f : 0.0f)")
}

pub fn not_equal_constant(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scalar: f32,
) -> Result<ArrayPtr> {
    apply_binary_math_operation(device, src, dst, scalar, "(x != y ? 1.0f : 0.0f)")
}
