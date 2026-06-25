use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

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
    let dst = tier0::create_like(src0, dst, DType::Unknown, device)?;
    let kernel = ("mask_label", include_str!("../../kernels/mask_label.cl"));
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar", ParameterValue::Float(label)),
    ];
    let range = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
