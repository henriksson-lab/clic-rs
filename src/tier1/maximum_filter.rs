use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, execute_separable, ParameterValue};
use crate::tier0;

/// Maximum filter with box (separable) or sphere connectivity.
pub fn maximum_filter(
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
    if connectivity == "sphere" {
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
        execute(
            device,
            (
                "maximum_sphere",
                include_str!("../../kernels/maximum_sphere.cl"),
            ),
            &params,
            global,
            [0, 0, 0],
            &[],
        )?;
    } else {
        let sigma = [radius_x, radius_y, radius_z];
        execute_separable(
            device,
            (
                "maximum_separable",
                include_str!("../../kernels/maximum_separable.cl"),
            ),
            src,
            &dst,
            sigma,
            r,
            [0, 0, 0],
        )?;
    }
    Ok(dst)
}

pub fn maximum_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    maximum_filter(device, src, dst, radius_x, radius_y, radius_z, "sphere")
}

pub fn maximum_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    maximum_filter(device, src, dst, radius_x, radius_y, radius_z, "box")
}
