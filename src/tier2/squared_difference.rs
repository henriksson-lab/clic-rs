use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{evaluate, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// (src0 - src1)^2 element-wise.
pub fn squared_difference(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src0, dst, DType::Float, device)?;
    evaluate(
        device,
        "pow(src0 - src1, 2)",
        &[
            ParameterValue::Array(src0.clone()),
            ParameterValue::Array(src1.clone()),
        ],
        &dst,
    )?;
    Ok(dst)
}
