use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

/// Flip an array along the given axes.
pub fn flip(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("flipx", ParameterValue::Int(flip_x as i32)),
        ("flipy", ParameterValue::Int(flip_y as i32)),
        ("flipz", ParameterValue::Int(flip_z as i32)),
    ];
    execute(
        device,
        ("flip", include_str!("../../kernels/flip.cl")),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
