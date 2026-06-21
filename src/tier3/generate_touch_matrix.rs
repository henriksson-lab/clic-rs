use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier2;
use crate::types::INDEX;

/// Generate a binary matrix indicating which labels touch each other.
///
/// Mirrors CLIc's `generate_touch_matrix_func`.
pub fn generate_touch_matrix(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst_matrix: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let max_label = tier2::maximum_of_all_pixels(device, src)? as usize + 1;
    let dst_matrix = tier0::create_dst(src, dst_matrix, max_label, max_label, 1, INDEX, device)?;
    dst_matrix.lock().unwrap().fill(0.0)?;

    let global = {
        let lock = src.lock().unwrap();
        [lock.width(), lock.height(), lock.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst_matrix.clone())),
    ];
    execute(
        device,
        (
            "generate_touch_matrix",
            include_str!("../../kernels/generate_touch_matrix.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;

    Ok(dst_matrix)
}
