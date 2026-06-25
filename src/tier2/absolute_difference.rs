use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{evaluate, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// |src0 - src1| element-wise.
pub fn absolute_difference(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src0, dst, DType::Unknown, device)?;
    evaluate(
        device,
        "fabs(src0 - src1)",
        &[
            ParameterValue::Array(src0.clone()),
            ParameterValue::Array(src1.clone()),
        ],
        &dst,
    )?;
    Ok(dst)
}
