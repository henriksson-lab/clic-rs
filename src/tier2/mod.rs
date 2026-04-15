//! Tier 2 — higher-level operations composing tier1 primitives.
//!
//! Mirrors CLIc's `clic/src/tier2/` directory.

use std::f32::consts::PI;

use crate::array::{pull, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{ConstantValue, ParameterValue, execute};
use crate::tier0;
use crate::tier1;
use crate::types::DType;

// ── Inline clip kernel ────────────────────────────────────────────────────────

const CLIP_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__kernel void clip_kernel(IMAGE_src_TYPE src, IMAGE_dst_TYPE dst, float lo, float hi) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    float v = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(clamp(v, lo, hi)));
}
"#;

// ── Simple composition functions ──────────────────────────────────────────────

/// |src0 - src1| element-wise.
pub fn absolute_difference(device: &DeviceArc, src0: &ArrayPtr, src1: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src0, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst",  ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![("APPLY_OP(x,y)", ConstantValue::Str("fabs(x - y)".to_string()))];
    execute(device, ("image_operation", crate::tier1::IMAGE_OPERATION_SRC), &params, global, [0, 0, 0], &constants)?;
    Ok(dst)
}

/// src0 + src1 element-wise.
pub fn add_images(device: &DeviceArc, src0: &ArrayPtr, src1: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    tier1::add_images_weighted(device, src0, src1, dst, 1.0, 1.0)
}

/// src0 - src1 element-wise.
pub fn subtract_images(device: &DeviceArc, src0: &ArrayPtr, src1: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    tier1::add_images_weighted(device, src0, src1, dst, 1.0, -1.0)
}

/// Negate: -src.
pub fn invert(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    tier1::multiply_image_and_scalar(device, src, dst, -1.0)
}

/// src^2.
pub fn square(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    tier1::power(device, src, dst, 2.0)
}

/// (src0 - src1)^2 element-wise.
pub fn squared_difference(device: &DeviceArc, src0: &ArrayPtr, src1: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src0, dst, DType::Float, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst",  ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![("APPLY_OP(x,y)", ConstantValue::Str("pow(x - y, 2.0f)".to_string()))];
    execute(device, ("image_operation", crate::tier1::IMAGE_OPERATION_SRC), &params, global, [0, 0, 0], &constants)?;
    Ok(dst)
}

/// Divide all pixels by a scalar.
pub fn divide_image_by_scalar(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, scalar: f32) -> Result<ArrayPtr> {
    tier1::divide_image_and_scalar(device, src, dst, scalar)
}

/// Convert degrees to radians: src * π / 180.
pub fn degrees_to_radians(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    tier1::multiply_image_and_scalar(device, src, Some(dst), PI / 180.0)
}

/// Convert radians to degrees: src * 180 / π.
pub fn radians_to_degrees(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    tier1::multiply_image_and_scalar(device, src, Some(dst), 180.0 / PI)
}

/// Clamp pixel values to [min_intensity, max_intensity].
pub fn clip(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, min_intensity: f32, max_intensity: f32) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("lo",  ParameterValue::Float(min_intensity)),
        ("hi",  ParameterValue::Float(max_intensity)),
    ];
    execute(device, ("clip_kernel", CLIP_SRC), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Gaussian(σ1) - Gaussian(σ2).
pub fn difference_of_gaussian(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    sigma1: [f32; 3],
    sigma2: [f32; 3],
) -> Result<ArrayPtr> {
    let [sigma1_x, sigma1_y, sigma1_z] = sigma1;
    let [sigma2_x, sigma2_y, sigma2_z] = sigma2;
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let g1 = tier1::gaussian_blur(device, src, None, sigma1_x, sigma1_y, sigma1_z)?;
    let g2 = tier1::gaussian_blur(device, src, None, sigma2_x, sigma2_y, sigma2_z)?;
    tier1::add_images_weighted(device, &g1, &g2, Some(dst), 1.0, -1.0)
}

// ── Global reductions ─────────────────────────────────────────────────────────

/// Return the maximum pixel value across the entire array.
pub fn maximum_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let (_, h, d) = { let l = src.lock().unwrap(); (l.width(), l.height(), l.depth()) };
    let mut tmp = src.clone();
    if d > 1 { tmp = tier1::maximum_projection(device, &tmp, None, 2)?; }
    if h > 1 { tmp = tier1::maximum_projection(device, &tmp, None, 1)?; }
    let dst = tier0::create_one(device)?;
    tier1::maximum_projection(device, &tmp, Some(dst.clone()), 0)?;
    let v: Vec<f32> = pull(&dst)?;
    Ok(v[0])
}

/// Return the minimum pixel value across the entire array.
pub fn minimum_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let (_, h, d) = { let l = src.lock().unwrap(); (l.width(), l.height(), l.depth()) };
    let mut tmp = src.clone();
    if d > 1 { tmp = tier1::minimum_projection(device, &tmp, None, 2)?; }
    if h > 1 { tmp = tier1::minimum_projection(device, &tmp, None, 1)?; }
    let dst = tier0::create_one(device)?;
    tier1::minimum_projection(device, &tmp, Some(dst.clone()), 0)?;
    let v: Vec<f32> = pull(&dst)?;
    Ok(v[0])
}

/// Return the sum of all pixel values.
pub fn sum_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let (_, h, d) = { let l = src.lock().unwrap(); (l.width(), l.height(), l.depth()) };
    let mut tmp = src.clone();
    if d > 1 { tmp = tier1::sum_projection(device, &tmp, None, 2)?; }
    if h > 1 { tmp = tier1::sum_projection(device, &tmp, None, 1)?; }
    let dst = tier0::create_one(device)?;
    tier1::sum_projection(device, &tmp, Some(dst.clone()), 0)?;
    let v: Vec<f32> = pull(&dst)?;
    Ok(v[0])
}

// ── Morphological operations ──────────────────────────────────────────────────

/// Grayscale closing (max filter then min filter) with box connectivity.
pub fn closing_box(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, rx: f32, ry: f32, rz: f32) -> Result<ArrayPtr> {
    let tmp = tier1::maximum_filter(device, src, None, rx, ry, rz, "box")?;
    tier1::minimum_filter(device, &tmp, dst, rx, ry, rz, "box")
}

/// Grayscale closing with sphere connectivity.
pub fn closing_sphere(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, rx: f32, ry: f32, rz: f32) -> Result<ArrayPtr> {
    let tmp = tier1::maximum_filter(device, src, None, rx, ry, rz, "sphere")?;
    tier1::minimum_filter(device, &tmp, dst, rx, ry, rz, "sphere")
}

/// Grayscale opening (min filter then max filter) with box connectivity.
pub fn opening_box(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, rx: f32, ry: f32, rz: f32) -> Result<ArrayPtr> {
    let tmp = tier1::minimum_filter(device, src, None, rx, ry, rz, "box")?;
    tier1::maximum_filter(device, &tmp, dst, rx, ry, rz, "box")
}

/// Grayscale opening with sphere connectivity.
pub fn opening_sphere(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, rx: f32, ry: f32, rz: f32) -> Result<ArrayPtr> {
    let tmp = tier1::minimum_filter(device, src, None, rx, ry, rz, "sphere")?;
    tier1::maximum_filter(device, &tmp, dst, rx, ry, rz, "sphere")
}

/// Top-hat: src - opening.
pub fn top_hat_box(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, rx: f32, ry: f32, rz: f32) -> Result<ArrayPtr> {
    let opened = opening_box(device, src, None, rx, ry, rz)?;
    tier1::add_images_weighted(device, src, &opened, dst, 1.0, -1.0)
}

/// Top-hat with sphere connectivity.
pub fn top_hat_sphere(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, rx: f32, ry: f32, rz: f32) -> Result<ArrayPtr> {
    let opened = opening_sphere(device, src, None, rx, ry, rz)?;
    tier1::add_images_weighted(device, src, &opened, dst, 1.0, -1.0)
}

/// Bottom-hat (black-hat): closing - src.
pub fn bottom_hat_box(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, rx: f32, ry: f32, rz: f32) -> Result<ArrayPtr> {
    let closed = closing_box(device, src, None, rx, ry, rz)?;
    tier1::add_images_weighted(device, &closed, src, dst, 1.0, -1.0)
}

/// Bottom-hat with sphere connectivity.
pub fn bottom_hat_sphere(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, rx: f32, ry: f32, rz: f32) -> Result<ArrayPtr> {
    let closed = closing_sphere(device, src, None, rx, ry, rz)?;
    tier1::add_images_weighted(device, &closed, src, dst, 1.0, -1.0)
}

/// Standard deviation filter with box connectivity.
pub fn standard_deviation_box(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, rx: f32, ry: f32, rz: f32) -> Result<ArrayPtr> {
    let var = tier1::variance_filter(device, src, None, rx, ry, rz, "box")?;
    tier1::power(device, &var, dst, 0.5)
}

/// Standard deviation filter with sphere connectivity.
pub fn standard_deviation_sphere(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>, rx: f32, ry: f32, rz: f32) -> Result<ArrayPtr> {
    let var = tier1::variance_filter(device, src, None, rx, ry, rz, "sphere")?;
    tier1::power(device, &var, dst, 0.5)
}
