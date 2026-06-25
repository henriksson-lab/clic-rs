use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Crop a substack from `src`.
///
/// Mirrors CLIc's `crop_func`.
#[allow(clippy::too_many_arguments)]
pub fn crop(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    start_x: i32,
    start_y: i32,
    start_z: i32,
    width: usize,
    height: usize,
    depth: usize,
) -> Result<ArrayPtr> {
    let dst = tier0::create_dst(src, dst, width, height, depth, DType::Unknown, device)?;
    let kernel = ("crop", include_str!("../../kernels/crop.cl"));
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index0", ParameterValue::Int(start_x)),
        ("index1", ParameterValue::Int(start_y)),
        ("index2", ParameterValue::Int(start_z)),
    ];
    let range = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
