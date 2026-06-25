use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Apply a Laplace filter with box or sphere/diamond connectivity.
///
/// Mirrors CLIc's `laplace_func`.
pub fn laplace(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let mut kernel = ("laplace_box", include_str!("../../kernels/laplace_box.cl"));
    if connectivity == "sphere" {
        kernel = (
            "laplace_diamond",
            include_str!("../../kernels/laplace_diamond.cl"),
        );
    }
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Apply a Laplace filter with box connectivity.
///
/// Mirrors CLIc's `laplace_box_func`.
pub fn laplace_box(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    laplace(device, src, dst, "box")
}

/// Apply a Laplace filter with sphere/diamond connectivity.
///
/// Mirrors CLIc's `laplace_diamond_func`.
pub fn laplace_diamond(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    laplace(device, src, dst, "sphere")
}
