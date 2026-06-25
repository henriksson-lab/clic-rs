use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier1;

pub fn opening(
    device: &DeviceArc,
    src: &ArrayPtr,
    footprint: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    if src.lock().unwrap().dimension() != footprint.lock().unwrap().dimension() {
        return Err(CleError::DimensionMismatch);
    }
    let temp = tier1::erosion(device, src, footprint, None)?;
    tier1::dilation(device, &temp, footprint, dst)
}

pub fn binary_opening(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let temp = tier1::binary_erode(
        device,
        src,
        None,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )?;
    tier1::binary_dilate(
        device,
        &temp,
        dst,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )
}

pub fn grayscale_opening(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let temp = tier1::minimum_filter(
        device,
        src,
        None,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )?;
    tier1::maximum_filter(
        device,
        &temp,
        dst,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )
}

/// Grayscale opening (min filter then max filter) with box connectivity.
pub fn opening_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    grayscale_opening(device, src, dst, radius_x, radius_y, radius_z, "box")
}

/// Grayscale opening with sphere connectivity.
pub fn opening_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    grayscale_opening(device, src, dst, radius_x, radius_y, radius_z, "sphere")
}
