use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Replace every pixel in `src0` by the lookup value from `src1` at that pixel index.
///
/// Mirrors CLIc's `replace_values_func`.
pub fn replace_values(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src0, dst, DType::Unknown, device)?;
    let kernel = (
        "replace_values",
        include_str!("../../kernels/replace_values.cl"),
    );
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Replace all pixels equal to `value_to_replace` with `value_replacement`.
///
/// Mirrors CLIc's `replace_value_func`.
pub fn replace_value(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    value_to_replace: f32,
    value_replacement: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let kernel = (
        "replace_value",
        include_str!("../../kernels/replace_value.cl"),
    );
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar0", ParameterValue::Float(value_to_replace)),
        ("scalar1", ParameterValue::Float(value_replacement)),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Alias for [`replace_value`].
///
/// Mirrors CLIc's `replace_intensity_func`.
pub fn replace_intensity(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    value_to_replace: f32,
    value_replacement: f32,
) -> Result<ArrayPtr> {
    replace_value(device, src, dst, value_to_replace, value_replacement)
}

/// Alias for [`replace_values`].
///
/// Mirrors CLIc's `replace_intensities_func`.
pub fn replace_intensities(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    replace_values(device, src0, src1, dst)
}
