use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier1;

pub fn closing(
    device: &DeviceArc,
    src: &ArrayPtr,
    footprint: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    if src.lock().unwrap().dimension() != footprint.lock().unwrap().dimension() {
        return Err(CleError::DimensionMismatch);
    }
    let temp = tier1::dilation(device, src, footprint, None)?;
    tier1::erosion(device, &temp, footprint, dst)
}

pub fn binary_closing(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let temp = tier1::binary_dilate(
        device,
        src,
        None,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )?;
    tier1::binary_erode(
        device,
        &temp,
        dst,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )
}

pub fn grayscale_closing(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let temp = tier1::maximum_filter(
        device,
        src,
        None,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )?;
    tier1::minimum_filter(
        device,
        &temp,
        dst,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )
}

/// Grayscale closing (max filter then min filter) with box connectivity.
pub fn closing_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    grayscale_closing(device, src, dst, radius_x, radius_y, radius_z, "box")
}

/// Grayscale closing with sphere connectivity.
pub fn closing_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    grayscale_closing(device, src, dst, radius_x, radius_y, radius_z, "sphere")
}
