use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::transform::{apply_affine_transform, AffineTransform};

/// Scales the image by given factors.
pub fn scale(
    _device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    factor_x: f32,
    factor_y: f32,
    factor_z: f32,
    centered: bool,
    interpolate: bool,
    resize: bool,
) -> Result<ArrayPtr> {
    let mut transform = AffineTransform::new();
    if centered && !resize {
        let shape = {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        };
        transform.center(shape, false);
    }
    transform.scale(factor_x, factor_y, factor_z);
    if centered && !resize {
        let shape = {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        };
        transform.center(shape, true);
    }
    apply_affine_transform(src, dst, &transform, interpolate, resize)
}
