use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier2;
use crate::types::INDEX;

/// Generate a binary label-overlap matrix between two label images.
///
/// Mirrors CLIc's `generate_binary_overlap_matrix_func`.
pub fn generate_binary_overlap_matrix(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let max_label_0 = tier2::maximum_of_all_pixels(device, src0)? as usize + 1;
    let max_label_1 = tier2::maximum_of_all_pixels(device, src1)? as usize + 1;
    let dst = tier0::create_dst(src0, dst, max_label_0, max_label_1, 1, INDEX, device)?;
    dst.lock().unwrap().fill(0.0)?;

    let global = {
        let lock = src0.lock().unwrap();
        [lock.width(), lock.height(), lock.depth()]
    };
    let params = vec![
        ("src0", ParameterValue::Array(src0.clone())),
        ("src1", ParameterValue::Array(src1.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        (
            "generate_binary_overlap_matrix",
            include_str!("../../kernels/generate_binary_overlap_matrix.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;

    Ok(dst)
}
