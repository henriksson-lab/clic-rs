//! Tier 7 — affine transform operations.
//!
//! Mirrors CLIc's `clic/src/tier7/` directory.
//! The `mat` buffer passed to the affine_transform kernel contains the flat
//! row-major representation of the 3×4 inverse transform matrix, i.e.
//! column-major of (M⁻¹)ᵀ (the Eigen convention used in CLIc).

use crate::array::{push, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{ParameterValue, execute};
use crate::tier0;

const AFFINE_TRANSFORM_SRC: &str = include_str!("../../kernels/affine_transform.cl");

/// Apply an affine transform to `src` using the given 4×4 inverse matrix
/// (row-major, top-3-rows only used by kernel as `mat[0..11]`).
///
/// `inv_mat_row_major` — 16 floats, row-major layout of M⁻¹:
///   mat[0..3]   = row 0,  mat[4..7] = row 1,  mat[8..11] = row 2
pub fn apply_affine_transform(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    inv_mat_row_major: &[f32; 16],
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = { let l = dst.lock().unwrap(); [l.width(), l.height(), l.depth()] };

    // Upload the 16-float matrix to GPU as a 1D float buffer (16×1×1).
    let mat = push::<f32>(inv_mat_row_major, 16, 1, 1, device)?;

    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("mat", ParameterValue::Array(mat)),
    ];
    execute(device, ("affine_transform", AFFINE_TRANSFORM_SRC), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Translate the image by (translate_x, translate_y, translate_z) pixels.
///
/// Positive values shift the image content towards higher indices.
pub fn translate(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    translate_x: f32,
    translate_y: f32,
    translate_z: f32,
) -> Result<ArrayPtr> {
    // Forward translation: new = old + t  →  inverse: old = new - t
    // Kernel reads src at (x - tx, y - ty, z - tz) for each dst pixel.
    #[rustfmt::skip]
    let inv: [f32; 16] = [
        1.0, 0.0, 0.0, -translate_x,
        0.0, 1.0, 0.0, -translate_y,
        0.0, 0.0, 1.0, -translate_z,
        0.0, 0.0, 0.0,  1.0,
    ];
    apply_affine_transform(device, src, dst, &inv)
}

/// Scale the image by (scale_x, scale_y, scale_z) around the origin.
///
/// A factor > 1 stretches; < 1 shrinks.
pub fn scale(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
) -> Result<ArrayPtr> {
    // Inverse of scale(sx,sy,sz) is scale(1/sx, 1/sy, 1/sz).
    #[rustfmt::skip]
    let inv: [f32; 16] = [
        1.0 / scale_x, 0.0,           0.0,           0.0,
        0.0,           1.0 / scale_y, 0.0,           0.0,
        0.0,           0.0,           1.0 / scale_z, 0.0,
        0.0,           0.0,           0.0,            1.0,
    ];
    apply_affine_transform(device, src, dst, &inv)
}
