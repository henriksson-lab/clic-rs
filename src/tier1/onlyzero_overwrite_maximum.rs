use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

/// Replace zero-valued pixels with the maximum neighboring value.
///
/// Writes `1` into `flag` when any pixel changes. Uses box connectivity by
/// default, or sphere/diamond connectivity when `connectivity == "sphere"`.
/// Mirrors CLIc's `onlyzero_overwrite_maximum_func`.
pub fn onlyzero_overwrite_maximum(
    device: &DeviceArc,
    src: &ArrayPtr,
    flag: &ArrayPtr,
    dst: Option<ArrayPtr>,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst0", ParameterValue::Array(flag.clone())),
        ("dst1", ParameterValue::Array(dst.clone())),
    ];
    let kernel = if connectivity == "sphere" {
        (
            "onlyzero_overwrite_maximum_diamond",
            include_str!("../../kernels/onlyzero_overwrite_maximum_diamond.cl"),
        )
    } else {
        (
            "onlyzero_overwrite_maximum_box",
            include_str!("../../kernels/onlyzero_overwrite_maximum_box.cl"),
        )
    };
    execute(device, kernel, &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Replace zero-valued pixels using sphere/diamond connectivity.
///
/// Mirrors CLIc's `onlyzero_overwrite_maximum_diamond_func`.
pub fn onlyzero_overwrite_maximum_diamond(
    device: &DeviceArc,
    src: &ArrayPtr,
    flag: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    onlyzero_overwrite_maximum(device, src, flag, dst, "sphere")
}

/// Replace zero-valued pixels using box connectivity.
///
/// Mirrors CLIc's `onlyzero_overwrite_maximum_box_func`.
pub fn onlyzero_overwrite_maximum_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    flag: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    onlyzero_overwrite_maximum(device, src, flag, dst, "box")
}
