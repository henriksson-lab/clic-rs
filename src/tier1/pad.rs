use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Extend `src` to an explicit shape, filling uncovered pixels with `value`.
///
/// When `center` is true, the source is placed using the same ceil-half offset
/// as CLIc's `pad_func`; otherwise it is placed at the origin.
pub fn pad(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    value: f32,
    center: bool,
) -> Result<ArrayPtr> {
    let (src_width, src_height, src_depth) = {
        let s = src.lock().unwrap();
        (s.width(), s.height(), s.depth())
    };
    let dst = tier0::create_dst(src, dst, size_x, size_y, size_z, DType::Unknown, device)?;
    dst.lock().unwrap().fill(value)?;

    let (offset_x, offset_y, offset_z) = pad_offset(
        src_width, src_height, src_depth, size_x, size_y, size_z, center,
    );
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar0", ParameterValue::Int(offset_x)),
        ("scalar1", ParameterValue::Int(offset_y)),
        ("scalar2", ParameterValue::Int(offset_z)),
    ];
    execute(
        device,
        ("paste", include_str!("../../kernels/paste.cl")),
        &params,
        [src_width, src_height, src_depth],
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

/// Extract an explicit shape from `src`, optionally centered like CLIc's `unpad_func`.
pub fn unpad(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    center: bool,
) -> Result<ArrayPtr> {
    let (src_width, src_height, src_depth) = {
        let s = src.lock().unwrap();
        (s.width(), s.height(), s.depth())
    };
    let dst = tier0::create_dst(src, dst, size_x, size_y, size_z, DType::Unknown, device)?;

    let (offset_x, offset_y, offset_z) = pad_offset(
        src_width, src_height, src_depth, size_x, size_y, size_z, center,
    );
    let global = {
        let d = dst.lock().unwrap();
        [d.width(), d.height(), d.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index0", ParameterValue::Int(offset_x)),
        ("index1", ParameterValue::Int(offset_y)),
        ("index2", ParameterValue::Int(offset_z)),
    ];
    execute(
        device,
        ("crop", include_str!("../../kernels/crop.cl")),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

fn pad_offset(
    src_width: usize,
    src_height: usize,
    src_depth: usize,
    dst_width: usize,
    dst_height: usize,
    dst_depth: usize,
    center: bool,
) -> (i32, i32, i32) {
    if !center {
        return (0, 0, 0);
    }

    (
        centered_offset(src_width, dst_width),
        centered_offset(src_height, dst_height),
        centered_offset(src_depth, dst_depth),
    )
}

fn centered_offset(src: usize, dst: usize) -> i32 {
    if dst > 1 {
        src.abs_diff(dst).div_ceil(2) as i32
    } else {
        0
    }
}
