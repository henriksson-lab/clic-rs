use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;
use crate::utils::shape_to_dimension;

/// Sum contiguous blocks along the active dimension selected from `src` shape.
///
/// Mirrors CLIc's `sum_reduction_x`: when `dst` is omitted, the destination
/// size is reduced by `blocksize` along x for 1D input, y for 2D input, and z
/// for 3D input. The upstream implementation notes that only 1D data was
/// tested.
pub fn sum_reduction_x(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    blocksize: i32,
) -> Result<ArrayPtr> {
    if blocksize <= 0 {
        return Err(CleError::Other("blocksize must be positive".into()));
    }

    let (mut width, mut height, mut depth) = {
        let s = src.lock().unwrap();
        (s.width(), s.height(), s.depth())
    };

    let blocksize_usize = blocksize as usize;
    match shape_to_dimension(width, height, depth) {
        1 => width /= blocksize_usize,
        2 => height /= blocksize_usize,
        3 => depth /= blocksize_usize,
        _ => {}
    }

    let dst = tier0::create_dst(src, dst, width, height, depth, DType::Unknown, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index", ParameterValue::Int(blocksize)),
    ];
    execute(
        device,
        (
            "sum_reduction_x",
            include_str!("../../kernels/sum_reduction_x.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
