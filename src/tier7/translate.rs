use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::transform::{apply_affine_transform, AffineTransform};

/// Translate the image by `(translate_x, translate_y, translate_z)` pixels.
///
/// Positive values shift the image content towards higher indices.
pub fn translate(
    _device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    translate_x: f32,
    translate_y: f32,
    translate_z: f32,
    interpolate: bool,
) -> Result<ArrayPtr> {
    let mut transform = AffineTransform::new();
    transform.translate(translate_x, translate_y, translate_z);
    apply_affine_transform(src, dst, &transform, interpolate, false)
}
