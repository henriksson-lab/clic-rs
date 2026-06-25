use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Flip an array along the given axes.
pub fn flip(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index0", ParameterValue::Int(flip_x as i32)),
        ("index1", ParameterValue::Int(flip_y as i32)),
        ("index2", ParameterValue::Int(flip_z as i32)),
    ];
    execute(
        device,
        ("flip", include_str!("../../kernels/flip.cl")),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
