use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Compute the x-axis gradient.
///
/// Mirrors CLIc's `gradient_x_func`.
pub fn gradient_x(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("axis", ParameterValue::Int(0)),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(
        device,
        ("gradient", include_str!("../../kernels/gradient.cl")),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

/// Compute the y-axis gradient.
///
/// Mirrors CLIc's `gradient_y_func`.
pub fn gradient_y(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("axis", ParameterValue::Int(1)),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(
        device,
        ("gradient", include_str!("../../kernels/gradient.cl")),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

/// Compute the z-axis gradient.
///
/// Mirrors CLIc's `gradient_z_func`.
pub fn gradient_z(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("axis", ParameterValue::Int(2)),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(
        device,
        ("gradient", include_str!("../../kernels/gradient.cl")),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
