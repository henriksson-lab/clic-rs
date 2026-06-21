use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
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
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![(
        "APPLY_OP(x,y)",
        ConstantValue::Str("pow(x - y, 2.0f)".to_string()),
    )];
    execute(
        device,
        ("image_operation", crate::tier1::IMAGE_OPERATION_SRC),
        &params,
        global,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}
