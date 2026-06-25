use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Copy a 2D image into a z slice of a 3D stack, or copy a z slice into a 2D image.
///
/// Mirrors CLIc's `copy_slice_func`.
pub fn copy_slice(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    slice_index: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index", ParameterValue::Int(slice_index)),
    ];
    let dst_depth = {
        let dst = dst.lock().unwrap();
        dst.depth()
    };
    let kernel;
    let range;
    if dst_depth > 1 {
        kernel = (
            "copy_slice_to",
            include_str!("../../kernels/copy_slice_to.cl"),
        );
        let src = src.lock().unwrap();
        range = [src.width(), src.height(), 1];
    } else {
        kernel = (
            "copy_slice_from",
            include_str!("../../kernels/copy_slice_from.cl"),
        );
        let dst = dst.lock().unwrap();
        range = [dst.width(), dst.height(), dst.depth()];
    }
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Copy a 2D image into a y slice of a 3D stack, or copy a y slice into a 2D image.
///
/// Mirrors CLIc's `copy_horizontal_slice_func`.
pub fn copy_horizontal_slice(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    slice_index: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index", ParameterValue::Int(slice_index)),
    ];
    let dst_depth = {
        let dst = dst.lock().unwrap();
        dst.depth()
    };
    let kernel;
    let range;
    if dst_depth > 1 {
        kernel = (
            "copy_horizontal_slice_to",
            include_str!("../../kernels/copy_horizontal_slice_to.cl"),
        );
        let dst = dst.lock().unwrap();
        range = [dst.width(), dst.height(), dst.depth()];
    } else {
        kernel = (
            "copy_horizontal_slice_from",
            include_str!("../../kernels/copy_horizontal_slice_from.cl"),
        );
        let dst = dst.lock().unwrap();
        range = [dst.width(), dst.height(), dst.depth()];
    }
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Copy a 2D image into an x slice of a 3D stack, or copy an x slice into a 2D image.
///
/// Mirrors CLIc's `copy_vertical_slice_func`.
pub fn copy_vertical_slice(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    slice_index: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index", ParameterValue::Int(slice_index)),
    ];
    let dst_depth = {
        let dst = dst.lock().unwrap();
        dst.depth()
    };
    let kernel;
    let range;
    if dst_depth > 1 {
        kernel = (
            "copy_vertical_slice_to",
            include_str!("../../kernels/copy_vertical_slice_to.cl"),
        );
        let src = src.lock().unwrap();
        range = [src.width(), src.height(), 1];
    } else {
        kernel = (
            "copy_vertical_slice_from",
            include_str!("../../kernels/copy_vertical_slice_from.cl"),
        );
        let dst = dst.lock().unwrap();
        range = [dst.width(), dst.height(), dst.depth()];
    }
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
