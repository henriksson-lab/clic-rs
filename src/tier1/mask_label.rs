use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

/// Copy pixels from `src0` where the corresponding `src1` label matches `label`.
///
/// Mirrors CLIc's `mask_label_func`.
pub fn mask_label(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
    label: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src0, dst, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar", ParameterValue::Float(label)),
    ];
    execute(
        device,
        ("mask_label", include_str!("../../kernels/mask_label.cl")),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
