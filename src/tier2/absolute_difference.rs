use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;

/// |src0 - src1| element-wise.
pub fn absolute_difference(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
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
    ];
    let constants = vec![(
        "APPLY_OP(x,y)",
        ConstantValue::Str("fabs(x - y)".to_string()),
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
