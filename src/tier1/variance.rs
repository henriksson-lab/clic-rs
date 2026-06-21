use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Variance filter with box or sphere connectivity.
pub fn variance_filter(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let r = [
        crate::utils::radius2kernelsize(radius_x),
        crate::utils::radius2kernelsize(radius_y),
        crate::utils::radius2kernelsize(radius_z),
    ];
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar0", ParameterValue::Int(r[0])),
        ("scalar1", ParameterValue::Int(r[1])),
        ("scalar2", ParameterValue::Int(r[2])),
    ];
    let (kname, ksrc) = if connectivity == "sphere" {
        (
            "variance_sphere",
            include_str!("../../kernels/variance_sphere.cl"),
        )
    } else {
        (
            "variance_box",
            include_str!("../../kernels/variance_box.cl"),
        )
    };
    execute(device, (kname, ksrc), &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

pub fn variance_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    variance_filter(device, src, dst, radius_x, radius_y, radius_z, "sphere")
}

pub fn variance_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    variance_filter(device, src, dst, radius_x, radius_y, radius_z, "box")
}
