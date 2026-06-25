use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::transform::{apply_affine_transform, AffineTransform};

/// Translates the image by a given vector and rotates it by given angles.
///
/// CLIc's C++ documentation currently says radians for this function, but the
/// implementation calls `AffineTransform::rotate`, which takes degrees. This
/// wrapper follows the implementation semantics: angles are given in degrees.
pub fn rigid_transform(
    _device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    translate_x: f32,
    translate_y: f32,
    translate_z: f32,
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
    transform.translate(translate_x, translate_y, translate_z);
    apply_affine_transform(src, dst, &transform, interpolate, resize)
}
