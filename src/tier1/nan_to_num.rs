use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Replace NaN and positive/negative infinity values with finite values.
///
/// Mirrors CLIc's `nan_to_num_func`.
pub fn nan_to_num(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    nan: f32,
    posinf: f32,
    neginf: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let kernel = ("nan_to_num", include_str!("../../kernels/nan_to_num.cl"));
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("nan", ParameterValue::Float(nan)),
        ("pinf", ParameterValue::Float(posinf)),
        ("ninf", ParameterValue::Float(neginf)),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
