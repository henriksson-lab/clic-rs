use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Enumerate labeled/non-zero values within blocks.
pub fn block_enumerate(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
    blocksize: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src0, dst, DType::Float, device)?;
    let global = {
        let src1 = src1.lock().unwrap();
        [src1.width(), src1.height(), src1.depth()]
    };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index", ParameterValue::Int(blocksize)),
    ];
    execute(
        device,
        (
            "block_enumerate",
            include_str!("../../kernels/block_enumerate.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
