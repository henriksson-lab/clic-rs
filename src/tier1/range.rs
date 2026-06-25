use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::utils::correct_range;

/// Crop an image according to explicit start/stop/step ranges.
///
/// Mirrors CLIc's `range_func`.
#[allow(clippy::too_many_arguments)]
pub fn range(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    start_x: i32,
    stop_x: i32,
    step_x: i32,
    start_y: i32,
    stop_y: i32,
    step_y: i32,
    start_z: i32,
    stop_z: i32,
    step_z: i32,
) -> Result<ArrayPtr> {
    let (src_width, src_height, src_depth, dtype) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.depth(), src.dtype())
    };

    let dst_width = ((stop_x - start_x).abs() / step_x.abs().max(1)) as usize;
    let dst_height = ((stop_y - start_y).abs() / step_y.abs().max(1)) as usize;
    let dst_depth = ((stop_z - start_z).abs() / step_z.abs().max(1)) as usize;
    let dst = tier0::create_dst(src, dst, dst_width, dst_height, dst_depth, dtype, device)?;

    let (start_x, _, step_x) =
        correct_range(Some(start_x), Some(stop_x), Some(step_x), src_width as i32);
    let (start_y, _, step_y) =
        correct_range(Some(start_y), Some(stop_y), Some(step_y), src_height as i32);
    let (start_z, _, step_z) =
        correct_range(Some(start_z), Some(stop_z), Some(step_z), src_depth as i32);

    let kernel = ("range", include_str!("../../kernels/range.cl"));
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("start_x", ParameterValue::Int(start_x)),
        ("step_x", ParameterValue::Int(step_x)),
        ("start_y", ParameterValue::Int(start_y)),
        ("step_y", ParameterValue::Int(step_y)),
        ("start_z", ParameterValue::Int(start_z)),
        ("step_z", ParameterValue::Int(step_z)),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
