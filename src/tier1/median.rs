use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;

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
    let dst = tier0::create_like_same(src, dst, device)?;
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
    let constants = vec![("MAX_ARRAY_SIZE", ConstantValue::Int(r[0] * r[1] * r[2]))];
    let (kname, ksrc) = if connectivity == "sphere" {
        (
            "median_sphere",
            include_str!("../../kernels/median_sphere.cl"),
        )
    } else {
        ("median_box", include_str!("../../kernels/median_box.cl"))
    };
    execute(
        device,
        (kname, ksrc),
        &params,
        global,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
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
