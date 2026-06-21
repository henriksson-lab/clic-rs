use super::affine_transform::affine_transform;
use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;

/// Translate the image by `(translate_x, translate_y, translate_z)` pixels.
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
    #[rustfmt::skip]
    let inv: [f32; 16] = [
        1.0, 0.0, 0.0, -translate_x,
        0.0, 1.0, 0.0, -translate_y,
        0.0, 0.0, 1.0, -translate_z,
        0.0, 0.0, 0.0,  1.0,
    ];
    affine_transform(device, src, dst, &inv)
}
