use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;
use crate::utils::radius2kernelsize;

/// Morphological dilation with an arbitrary binary footprint.
pub fn dilation(
    device: &DeviceArc,
    src: &ArrayPtr,
    footprint: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let src_dim = src.lock().unwrap().dimension();
    let footprint_dim = footprint.lock().unwrap().dimension();
    if src_dim != footprint_dim {
        return Err(CleError::Other(
            "Error: input and structuring element in dilation operator must have the same dimensionality.".into(),
        ));
    }
    let kernel = ("dilation", include_str!("../../kernels/dilation.cl"));
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("footprint", ParameterValue::Array(footprint.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Binary dilation with box or sphere connectivity.
pub fn binary_dilate(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let r_x = radius2kernelsize(radius_x);
    let r_y = radius2kernelsize(radius_y);
    let r_z = radius2kernelsize(radius_z);
    let mut kernel = ("dilate_box", include_str!("../../kernels/dilate_box.cl"));
    if connectivity == "sphere" {
        kernel = (
            "dilate_sphere",
            include_str!("../../kernels/dilate_sphere.cl"),
        );
    }
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
    Ok(dst)
}

/// Grayscale dilation with box or sphere connectivity.
pub fn grayscale_dilate(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    crate::tier1::maximum_filter(device, src, dst, radius_x, radius_y, radius_z, connectivity)
}

/// Morphological box dilation.
pub fn dilate_box(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    binary_dilate(device, src, dst, 1.0, 1.0, 1.0, "box")
}

/// Morphological sphere (cross) dilation.
pub fn dilate_sphere(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    binary_dilate(device, src, dst, 1.0, 1.0, 1.0, "sphere")
}
