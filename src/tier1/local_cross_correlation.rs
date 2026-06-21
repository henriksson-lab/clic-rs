use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Compute local normalized cross-correlation between an image and a kernel.
///
/// Mirrors CLIc's `local_cross_correlation_func`.
pub fn local_cross_correlation(
    device: &DeviceArc,
    src: &ArrayPtr,
    kernel: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src0", ParameterValue::Array(src.clone())),
        ("src1", ParameterValue::Array(kernel.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        (
            "local_cross_correlation",
            include_str!("../../kernels/local_cross_correlation.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
