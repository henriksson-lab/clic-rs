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
    distance_matrix_destination: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let width = coordinate_list1.lock().unwrap().width() + 1;
    let distance_matrix_destination = tier0::create_dst(
        coordinate_list1,
        distance_matrix_destination,
        width,
        width,
        1,
        DType::Float,
        device,
    )?;
    distance_matrix_destination.lock().unwrap().fill(0.0)?;
    let kernel = (
        "generate_distance_matrix",
        include_str!("../../kernels/generate_distance_matrix.cl"),
    );
    let params = vec![
        ("src0", ParameterValue::Array(coordinate_list1.clone())),
        ("src1", ParameterValue::Array(coordinate_list2.clone())),
        (
            "dst",
            ParameterValue::Array(distance_matrix_destination.clone()),
        ),
    ];
    let range = {
        let l = distance_matrix_destination.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(distance_matrix_destination)
}
