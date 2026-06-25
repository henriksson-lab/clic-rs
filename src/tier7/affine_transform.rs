//! Affine transform support.

use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::transform::{apply_affine_transform, AffineTransform};

/// Apply an affine transform to `src` using a 3x3 or 4x4 row-major matrix.
pub fn affine_transform(
    _device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    transform_matrix: Option<&[f32]>,
    interpolate: bool,
    resize: bool,
) -> Result<ArrayPtr> {
    let transform_matrix = transform_matrix.unwrap_or(&[
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]);
    if transform_matrix.len() != 16 && transform_matrix.len() != 9 {
        return Err(CleError::Other(
            "Error: Transformation matrix size must be 9 or 16.".to_string(),
        ));
    }

    let transform_matrix_arr = if transform_matrix.len() == 9 {
        [
            transform_matrix[0],
            transform_matrix[1],
            0.0,
            transform_matrix[2],
            transform_matrix[3],
            transform_matrix[4],
            0.0,
            transform_matrix[5],
            transform_matrix[6],
            transform_matrix[7],
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    } else {
        let mut transform_matrix_arr = [0.0; 16];
        transform_matrix_arr.copy_from_slice(transform_matrix);
        transform_matrix_arr
    };

    let transform = AffineTransform::from_array(transform_matrix_arr);
    apply_affine_transform(src, dst, &transform, interpolate, resize)
}
