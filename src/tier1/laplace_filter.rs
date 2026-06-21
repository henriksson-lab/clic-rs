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
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let kernel = if connectivity == "sphere" {
        (
            "laplace_diamond",
            include_str!("../../kernels/laplace_diamond.cl"),
        )
    } else {
        ("laplace_box", include_str!("../../kernels/laplace_box.cl"))
    };
    execute(device, kernel, &params, global, [0, 0, 0], &[])?;
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
