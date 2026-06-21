use super::affine_transform::affine_transform;
use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;

/// Scale the image by `(scale_x, scale_y, scale_z)` around the origin.
///
/// A factor greater than 1 stretches; a factor less than 1 shrinks.
pub fn scale(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
) -> Result<ArrayPtr> {
    #[rustfmt::skip]
    let inv: [f32; 16] = [
        1.0 / scale_x, 0.0,           0.0,           0.0,
        0.0,           1.0 / scale_y, 0.0,           0.0,
        0.0,           0.0,           1.0 / scale_z, 0.0,
        0.0,           0.0,           0.0,            1.0,
    ];
    affine_transform(device, src, dst, &inv)
}
