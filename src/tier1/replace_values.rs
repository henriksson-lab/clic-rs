use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

fn global_from(arr: &ArrayPtr) -> [usize; 3] {
    let l = arr.lock().unwrap();
    [l.width(), l.height(), l.depth()]
}

/// Replace every pixel in `src0` by the lookup value from `src1` at that pixel index.
///
/// Mirrors CLIc's `replace_values_func`.
pub fn replace_values(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src0, dst, device)?;
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        (
            "replace_values",
            include_str!("../../kernels/replace_values.cl"),
        ),
        &params,
        global_from(&dst),
        [0, 0, 0],
        &[],
    )?;
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
    let dst = tier0::create_like_same(src, dst, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar0", ParameterValue::Float(value_to_replace)),
        ("scalar1", ParameterValue::Float(value_replacement)),
    ];
    execute(
        device,
        (
            "replace_value",
            include_str!("../../kernels/replace_value.cl"),
        ),
        &params,
        global_from(&dst),
        [0, 0, 0],
        &[],
    )?;
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
