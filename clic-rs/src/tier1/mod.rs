//! Tier 1 — elementary GPU operations (mirrors CLIc's `tier1/`).
//!
//! Math operations (unary, trig, binary) are generated via `macro_rules!`
//! to eliminate the ~53% boilerplate identified in the C++ codebase.

use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, execute_separable, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::DType;

// ── Shared kernel sources ────────────────────────────────────────────────────

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

// ── Macro: unary math operation ──────────────────────────────────────────────

macro_rules! unary_math_op {
    ($fn_name:ident, $op_expr:literal) => {
        pub fn $fn_name(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
            let dst = tier0::create_like_same(src, dst, device)?;
            let global = {
                let lock = dst.lock().unwrap();
                [lock.width(), lock.height(), lock.depth()]
            };
            let params = vec![
                ("src", ParameterValue::Array(src.clone())),
                ("dst", ParameterValue::Array(dst.clone())),
            ];
            let constants = vec![
                ("OP(x)", ConstantValue::Str($op_expr.to_string())),
            ];
            execute(device, ("math_unary", MATH_UNARY_SRC), &params, global, [0, 0, 0], &constants)?;
            Ok(dst)
        }
    };
}

// ── Unary math operations ────────────────────────────────────────────────────

unary_math_op!(absolute,      "fabs(x)");
unary_math_op!(cubic_root,    "cbrt(x)");
unary_math_op!(square_root,   "sqrt(x)");
unary_math_op!(exponential,   "exp(x)");
unary_math_op!(exponential2,  "exp2(x)");
unary_math_op!(exponential10, "exp10(x)");
unary_math_op!(logarithm,     "log(x)");
unary_math_op!(logarithm2,    "log2(x)");
unary_math_op!(logarithm10,   "log10(x)");
unary_math_op!(reciprocal,    "1.0f / x");
unary_math_op!(ceil,          "ceil(x)");
unary_math_op!(floor,         "floor(x)");
unary_math_op!(round,         "round(x)");
unary_math_op!(truncate,      "trunc(x)");
unary_math_op!(binary_not,    "(x != 0) ? 0 : 1");
unary_math_op!(sign,          "(x > 0) ? 1 : ((x < 0) ? -1 : 0)");

// ── Trig operations (same pattern as unary math) ─────────────────────────────

unary_math_op!(sin_func,  "sin(x)");
unary_math_op!(cos_func,  "cos(x)");
unary_math_op!(tan_func,  "tan(x)");
unary_math_op!(asin_func, "asin(x)");
unary_math_op!(acos_func, "acos(x)");
unary_math_op!(atan_func, "atan(x)");
unary_math_op!(sinh_func, "sinh(x)");
unary_math_op!(cosh_func, "cosh(x)");
unary_math_op!(tanh_func, "tanh(x)");
unary_math_op!(asinh_func, "asinh(x)");
unary_math_op!(acosh_func, "acosh(x)");
unary_math_op!(atanh_func, "atanh(x)");

// ── Binary math operations (scalar) ─────────────────────────────────────────

macro_rules! binary_scalar_op {
    ($fn_name:ident, $op_expr:literal) => {
        pub fn $fn_name(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, scalar: f32) -> Result<ArrayPtr> {
            let dst = tier0::create_like_same(src, dst, device)?;
            let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
            let params = vec![
                ("src", ParameterValue::Array(src.clone())),
                ("dst", ParameterValue::Array(dst.clone())),
                ("scalar", ParameterValue::Float(scalar)),
            ];
            let constants = vec![("OP(x,y)", ConstantValue::Str($op_expr.to_string()))];
            execute(device, ("math_binary_scalar", MATH_BINARY_SCALAR_SRC), &params, global, [0, 0, 0], &constants)?;
            Ok(dst)
        }
    };
}

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

binary_scalar_op!(add_image_and_scalar,      "x + y");
binary_scalar_op!(subtract_image_and_scalar, "x - y");
binary_scalar_op!(multiply_image_and_scalar, "x * y");
binary_scalar_op!(divide_image_and_scalar,   "x / y");
binary_scalar_op!(power,                     "pow(x, y)");
binary_scalar_op!(greater_constant,          "(x > y) ? 1 : 0");
binary_scalar_op!(greater_or_equal_constant, "(x >= y) ? 1 : 0");
binary_scalar_op!(smaller_constant,          "(x < y) ? 1 : 0");
binary_scalar_op!(smaller_or_equal_constant, "(x <= y) ? 1 : 0");
binary_scalar_op!(equal_constant,            "(x == y) ? 1 : 0");
binary_scalar_op!(not_equal_constant,        "(x != y) ? 1 : 0");

// ── Two-image operations ─────────────────────────────────────────────────────

/// Add two images element-wise with independent scale factors.
pub fn add_images_weighted(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
    factor0: f32,
    factor1: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src0, dst, DType::Float, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst",  ParameterValue::Array(dst.clone())),
        ("scalar0", ParameterValue::Float(factor0)),
        ("scalar1", ParameterValue::Float(factor1)),
    ];
    execute(device, ("add_images_weighted", include_str!("../../kernels/add_images_weighted.cl")), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Element-wise maximum of two images.
pub fn maximum_images(device: &DeviceArc, src0: &ArrayPtr, src1: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src0, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst",  ParameterValue::Array(dst.clone())),
    ];
    execute(device, ("superior_inferior_2d", include_str!("../../kernels/superior_inferior_2d.cl")), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Element-wise minimum of two images.
pub fn minimum_images(device: &DeviceArc, src0: &ArrayPtr, src1: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src0, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst",  ParameterValue::Array(dst.clone())),
    ];
    execute(device, ("inferior_superior_2d", include_str!("../../kernels/inferior_superior_2d.cl")), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Copy src to dst.
pub fn copy(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(device, ("copy", include_str!("../../kernels/copy.cl")), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Gaussian blur with per-axis sigma values.
pub fn gaussian_blur(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    sigma_x: f32,
    sigma_y: f32,
    sigma_z: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;

    // Convert to float if needed
    let src_float = if src.lock().unwrap().dtype() != DType::Float {
        let t = tier0::create_like(src, None, DType::Float, device)?;
        copy(device, src, Some(t.clone()))?
    } else {
        src.clone()
    };

    let sigma = [sigma_x, sigma_y, sigma_z];
    let radius = sigma.map(|s| crate::utils::sigma2kernelsize(s));
    execute_separable(
        device,
        ("gaussian_blur_separable", include_str!("../../kernels/gaussian_blur_separable.cl")),
        &src_float,
        &dst,
        sigma,
        radius,
        [0, 0, 0],
    )?;
    Ok(dst)
}

/// Maximum projection along an axis (0=X, 1=Y, 2=Z).
pub fn maximum_projection(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, axis: usize) -> Result<ArrayPtr> {
    let dst = dst.unwrap_or(tier0::create_projection(src, axis, device)?);
    let global = { let l = src.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let (kernel_name, kernel_src) = match axis {
        0 => ("maximum_x_projection", include_str!("../../kernels/maximum_projection.cl")),
        1 => ("maximum_y_projection", include_str!("../../kernels/maximum_projection.cl")),
        _ => ("maximum_z_projection", include_str!("../../kernels/maximum_projection.cl")),
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(device, (kernel_name, kernel_src), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Minimum projection along an axis.
pub fn minimum_projection(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, axis: usize) -> Result<ArrayPtr> {
    let dst = dst.unwrap_or(tier0::create_projection(src, axis, device)?);
    let global = { let l = src.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let (kernel_name, kernel_src) = match axis {
        0 => ("minimum_x_projection", include_str!("../../kernels/minimum_projection.cl")),
        1 => ("minimum_y_projection", include_str!("../../kernels/minimum_projection.cl")),
        _ => ("minimum_z_projection", include_str!("../../kernels/minimum_projection.cl")),
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(device, (kernel_name, kernel_src), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Sum projection along an axis.
pub fn sum_projection(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, axis: usize) -> Result<ArrayPtr> {
    let dst = dst.unwrap_or(tier0::create_projection(src, axis, device)?);
    let global = { let l = src.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let (kernel_name, kernel_src) = match axis {
        0 => ("sum_x_projection", include_str!("../../kernels/sum_projection.cl")),
        1 => ("sum_y_projection", include_str!("../../kernels/sum_projection.cl")),
        _ => ("sum_z_projection", include_str!("../../kernels/sum_projection.cl")),
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(device, (kernel_name, kernel_src), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Mean projection along an axis.
pub fn mean_projection(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, axis: usize) -> Result<ArrayPtr> {
    let dst = dst.unwrap_or(tier0::create_projection(src, axis, device)?);
    let global = { let l = src.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let (kernel_name, kernel_src) = match axis {
        0 => ("mean_x_projection", include_str!("../../kernels/mean_projection.cl")),
        1 => ("mean_y_projection", include_str!("../../kernels/mean_projection.cl")),
        _ => ("mean_z_projection", include_str!("../../kernels/mean_projection.cl")),
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(device, (kernel_name, kernel_src), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Morphological box dilation.
pub fn dilate_box(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(device, ("dilate_box", include_str!("../../kernels/dilate_box.cl")), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Morphological box erosion.
pub fn erode_box(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(device, ("erode_box", include_str!("../../kernels/erode_box.cl")), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Morphological sphere (cross) dilation.
pub fn dilate_sphere(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(device, ("dilate_sphere", include_str!("../../kernels/dilate_sphere.cl")), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Morphological sphere (cross) erosion.
pub fn erode_sphere(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(device, ("erode_sphere", include_str!("../../kernels/erode_sphere.cl")), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Flip an array along the given axes.
pub fn flip(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, flip_x: bool, flip_y: bool, flip_z: bool) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("flipx", ParameterValue::Int(flip_x as i32)),
        ("flipy", ParameterValue::Int(flip_y as i32)),
        ("flipz", ParameterValue::Int(flip_z as i32)),
    ];
    execute(device, ("flip", include_str!("../../kernels/flip.cl")), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Apply a mask: set pixels to 0 where mask == 0.
pub fn mask(device: &DeviceArc, src: &ArrayPtr, mask_arr: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("mask", ParameterValue::Array(mask_arr.clone())),
        ("dst",  ParameterValue::Array(dst.clone())),
    ];
    execute(device, ("mask", include_str!("../../kernels/mask.cl")), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Set all pixels to a constant value.
pub fn set(device: &DeviceArc, arr: &ArrayPtr, value: f32) -> Result<()> {
    arr.lock().unwrap().fill(value)
}
