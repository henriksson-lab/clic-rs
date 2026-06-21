use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Generate a pairwise point-list distance matrix with CLIc's one-pixel zero border.
///
/// Mirrors CLIc's `generate_distance_matrix_func`.
pub fn generate_distance_matrix(
    device: &DeviceArc,
    coordinate_list1: &ArrayPtr,
    coordinate_list2: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let width = coordinate_list1.lock().unwrap().width() + 1;
    let dst = tier0::create_dst(coordinate_list1, dst, width, width, 1, DType::Float, device)?;
    dst.lock().unwrap().fill(0.0)?;

    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src0", ParameterValue::Array(coordinate_list1.clone())),
        ("src1", ParameterValue::Array(coordinate_list2.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        (
            "generate_distance_matrix",
            include_str!("../../kernels/generate_distance_matrix.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
