use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

use super::opening::{opening_box, opening_sphere};

pub fn top_hat(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let opened = if connectivity == "sphere" {
        opening_sphere(device, src, None, rx, ry, rz)?
    } else {
        opening_box(device, src, None, rx, ry, rz)?
    };
    tier1::add_images_weighted(device, src, &opened, dst, 1.0, -1.0)
}

/// Top-hat: src - opening.
pub fn top_hat_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
) -> Result<ArrayPtr> {
    top_hat(device, src, dst, rx, ry, rz, "box")
}

/// Top-hat with sphere connectivity.
pub fn top_hat_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    rx: f32,
    ry: f32,
    rz: f32,
) -> Result<ArrayPtr> {
    top_hat(device, src, dst, rx, ry, rz, "sphere")
}
