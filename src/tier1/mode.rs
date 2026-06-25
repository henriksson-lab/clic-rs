use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Mode filter with box or sphere connectivity.
pub fn mode(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    {
        let src = src.lock().unwrap();
        if src.dtype() != DType::Uint8 {
            eprintln!("Warning: mode only support uint8 pixel type.");
        }
    }

    let dst = tier0::create_like(src, dst, DType::Uint8, device)?;
    let r_x = crate::utils::radius2kernelsize(radius_x);
    let r_y = crate::utils::radius2kernelsize(radius_y);
    let r_z = crate::utils::radius2kernelsize(radius_z);
    let mut kernel = ("mode_box", include_str!("../../kernels/mode_box.cl"));
    if connectivity == "sphere" {
        kernel = ("mode_sphere", include_str!("../../kernels/mode_sphere.cl"));
    }
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar0", ParameterValue::Int(r_x)),
        ("scalar1", ParameterValue::Int(r_y)),
        ("scalar2", ParameterValue::Int(r_z)),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };

    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Mode filter using spherical connectivity.
pub fn mode_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    mode(device, src, dst, radius_x, radius_y, radius_z, "sphere")
}

/// Mode filter using box connectivity.
pub fn mode_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    mode(device, src, dst, radius_x, radius_y, radius_z, "box")
}
