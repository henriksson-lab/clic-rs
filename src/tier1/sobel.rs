use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Compute the Sobel gradient magnitude into a floating-point image.
///
/// Mirrors CLIc's `sobel_func`.
pub fn sobel(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        ("sobel", include_str!("../../kernels/sobel.cl")),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
