//! Affine transform support.
//!
//! The `mat` buffer passed to the affine_transform kernel contains the flat
//! row-major representation of the 3x4 inverse transform matrix, i.e.
//! column-major of `(M^-1)^T` in the Eigen convention used by CLIc.

use crate::array::{push, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

const AFFINE_TRANSFORM_SRC: &str = include_str!("../../kernels/affine_transform.cl");

/// Apply an affine transform to `src` using the given 4x4 inverse matrix.
///
/// `inv_mat_row_major` is 16 floats in row-major layout of `M^-1`. The kernel
/// reads the first three rows.
pub fn affine_transform(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    inv_mat_row_major: &[f32; 16],
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };

    let mat = push::<f32>(inv_mat_row_major, 16, 1, 1, device)?;

    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("mat", ParameterValue::Array(mat)),
    ];
    execute(
        device,
        ("affine_transform", AFFINE_TRANSFORM_SRC),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
