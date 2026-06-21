use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

/// Apply a non-zero maximum filter with box or sphere connectivity.
///
/// Mirrors CLIc's `nonzero_maximum_func`.
pub fn nonzero_maximum(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst0: &ArrayPtr,
    dst1: Option<ArrayPtr>,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let dst1 = tier0::create_like_same(src, dst1, device)?;
    let global = {
        let l = dst1.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst0", ParameterValue::Array(dst0.clone())),
        ("dst1", ParameterValue::Array(dst1.clone())),
    ];
    let kernel = if connectivity == "sphere" {
        (
            "nonzero_maximum_diamond",
            include_str!("../../kernels/nonzero_maximum_diamond.cl"),
        )
    } else {
        (
            "nonzero_maximum_box",
            include_str!("../../kernels/nonzero_maximum_box.cl"),
        )
    };
    execute(device, kernel, &params, global, [0, 0, 0], &[])?;
    Ok(dst1)
}

/// Apply a non-zero maximum filter with sphere connectivity.
///
/// Mirrors CLIc's `nonzero_maximum_diamond_func`.
pub fn nonzero_maximum_diamond(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst0: &ArrayPtr,
    dst1: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    nonzero_maximum(device, src, dst0, dst1, "sphere")
}

/// Apply a non-zero maximum filter with box connectivity.
///
/// Mirrors CLIc's `nonzero_maximum_box_func`.
pub fn nonzero_maximum_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst0: &ArrayPtr,
    dst1: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    nonzero_maximum(device, src, dst0, dst1, "box")
}
