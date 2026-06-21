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
    let r = [
        crate::utils::radius2kernelsize(radius_x),
        crate::utils::radius2kernelsize(radius_y),
        crate::utils::radius2kernelsize(radius_z),
    ];
    let global = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar0", ParameterValue::Int(r[0])),
        ("scalar1", ParameterValue::Int(r[1])),
        ("scalar2", ParameterValue::Int(r[2])),
    ];
    let kernel = if connectivity == "sphere" {
        ("mode_sphere", include_str!("../../kernels/mode_sphere.cl"))
    } else {
        ("mode_box", include_str!("../../kernels/mode_box.cl"))
    };

    execute(device, kernel, &params, global, [0, 0, 0], &[])?;
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
