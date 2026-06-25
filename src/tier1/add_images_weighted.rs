use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Add two images element-wise with independent scale factors.
pub fn add_images_weighted(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
    factor1: f32,
    factor2: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src0, dst, DType::Float, device)?;
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar0", ParameterValue::Float(factor1)),
        ("scalar1", ParameterValue::Float(factor2)),
    ];
    execute(
        device,
        (
            "add_images_weighted",
            include_str!("../../kernels/add_images_weighted.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
