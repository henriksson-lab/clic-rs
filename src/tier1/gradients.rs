use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

fn gradient_axis(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    axis: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("axis", ParameterValue::Int(axis)),
    ];
    execute(
        device,
        ("gradient", include_str!("../../kernels/gradient.cl")),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

/// Compute the x-axis gradient.
///
/// Mirrors CLIc's `gradient_x_func`.
pub fn gradient_x(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    gradient_axis(device, src, dst, 0)
}

/// Compute the y-axis gradient.
///
/// Mirrors CLIc's `gradient_y_func`.
pub fn gradient_y(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    gradient_axis(device, src, dst, 1)
}

/// Compute the z-axis gradient.
///
/// Mirrors CLIc's `gradient_z_func`.
pub fn gradient_z(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    gradient_axis(device, src, dst, 2)
}
