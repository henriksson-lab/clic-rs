use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

use super::closing::{closing_box, closing_sphere};

pub fn bottom_hat(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let closed = if connectivity == "sphere" {
        closing_sphere(device, src, None, rx, ry, rz)?
    } else {
        closing_box(device, src, None, rx, ry, rz)?
    };
    tier1::add_images_weighted(device, &closed, src, dst, 1.0, -1.0)
}

/// Bottom-hat (black-hat): closing - src.
pub fn bottom_hat_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
) -> Result<ArrayPtr> {
    bottom_hat(device, src, dst, rx, ry, rz, "box")
}

/// Bottom-hat with sphere connectivity.
pub fn bottom_hat_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
) -> Result<ArrayPtr> {
    bottom_hat(device, src, dst, rx, ry, rz, "sphere")
}
