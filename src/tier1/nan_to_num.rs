use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

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
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("nan", ParameterValue::Float(nan)),
        ("pinf", ParameterValue::Float(posinf)),
        ("ninf", ParameterValue::Float(neginf)),
    ];
    execute(
        device,
        ("nan_to_num", include_str!("../../kernels/nan_to_num.cl")),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
