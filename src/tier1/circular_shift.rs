use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

/// Apply a circular shift (roll) to `src`, wrapping border pixels per dimension.
pub fn circular_shift(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    shift_x: i32,
    shift_y: i32,
    shift_z: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index_1", ParameterValue::Int(shift_x)),
        ("index_2", ParameterValue::Int(shift_y)),
        ("index_3", ParameterValue::Int(shift_z)),
    ];
    execute(
        device,
        (
            "circular_shift",
            include_str!("../../kernels/circular_shift.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
