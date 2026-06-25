use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::transform::{apply_affine_transform, AffineTransform};

/// Rotates the image by given angles.
///
/// Angles are given in degrees. To convert radians to degrees, use this
/// formula: `angle_in_degrees = angle_in_radians * 180.0 / PI`.
pub fn rotate(
    _device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    angle_x: f32,
    angle_y: f32,
    angle_z: f32,
    centered: bool,
    interpolate: bool,
    resize: bool,
) -> Result<ArrayPtr> {
    let mut transform = AffineTransform::new();
    if centered {
        transform.center(
            {
                let src = src.lock().unwrap();
                [src.width(), src.height(), src.depth()]
            },
            false,
        );
    }
    if angle_x != 0.0 {
        transform.rotate(0, angle_x);
    }
    if angle_y != 0.0 {
        transform.rotate(1, angle_y);
    }
    if angle_z != 0.0 {
        transform.rotate(2, angle_z);
    }
    if centered {
        transform.center(
            {
                let src = src.lock().unwrap();
                [src.width(), src.height(), src.depth()]
            },
            true,
        );
    }
    apply_affine_transform(src, dst, &transform, interpolate, resize)
}
