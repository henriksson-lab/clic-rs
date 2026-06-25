use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Median filter with box or sphere connectivity.
pub fn median(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let r_x = crate::utils::radius2kernelsize(radius_x);
    let r_y = crate::utils::radius2kernelsize(radius_y);
    let r_z = crate::utils::radius2kernelsize(radius_z);
    let median_size = r_x * r_y * r_z;
    let item_size = src.lock().unwrap().item_size();
    if median_size as usize * item_size > device.get_local_memory_size() {
        eprintln!(
            "Warning: The kernel size is too large for the device local memory. Total kernel size is {} bytes, but the device local memory size is {} bytes.",
            median_size as usize * item_size,
            device.get_local_memory_size()
        );
    }
    let mut kernel = ("median_box", include_str!("../../kernels/median_box.cl"));
    if connectivity == "sphere" {
        kernel = (
            "median_sphere",
            include_str!("../../kernels/median_sphere.cl"),
        );
    }
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar0", ParameterValue::Int(r_x)),
        ("scalar1", ParameterValue::Int(r_y)),
        ("scalar2", ParameterValue::Int(r_z)),
    ];
    let constants = vec![("MAX_ARRAY_SIZE", ConstantValue::Int(r_x * r_y * r_z))];
    let range = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &constants)?;
    Ok(dst)
}

/// Median filter using box connectivity.
pub fn median_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    median(device, src, dst, radius_x, radius_y, radius_z, "box")
}

/// Median filter using spherical connectivity.
pub fn median_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    median(device, src, dst, radius_x, radius_y, radius_z, "sphere")
}
