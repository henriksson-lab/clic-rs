use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, execute_separable, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Mean filter with box (separable) or sphere connectivity.
pub fn mean_filter(
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
    if connectivity == "sphere" {
        let kernel = ("mean_sphere", include_str!("../../kernels/mean_sphere.cl"));
        let params = vec![
            ("src", ParameterValue::Array(src.clone())),
            ("dst", ParameterValue::Array(dst.clone())),
            ("scalar0", ParameterValue::Int(r_x)),
            ("scalar1", ParameterValue::Int(r_y)),
            ("scalar2", ParameterValue::Int(r_z)),
        ];
        let range = {
            let l = dst.lock().unwrap();
            [l.width(), l.height(), l.depth()]
        };
        execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    } else {
        let kernel = (
            "mean_separable",
            include_str!("../../kernels/mean_separable.cl"),
        );
        let sigma = [radius_x, radius_y, radius_z];
        execute_separable(device, kernel, src, &dst, sigma, [r_x, r_y, r_z], [0, 0, 0])?;
    }
    Ok(dst)
}

/// Mean filter using spherical connectivity.
pub fn mean_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    mean_filter(device, src, dst, radius_x, radius_y, radius_z, "sphere")
}

/// Mean filter using box connectivity.
pub fn mean_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    mean_filter(device, src, dst, radius_x, radius_y, radius_z, "box")
}
