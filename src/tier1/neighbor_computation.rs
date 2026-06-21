use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::DType;

fn touching_neighbors_op(
    device: &DeviceArc,
    vector: &ArrayPtr,
    matrix: &ArrayPtr,
    dst: Option<ArrayPtr>,
    kernel: (&'static str, &'static str),
    constants: &[(&str, ConstantValue)],
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(vector, dst, DType::Float, device)?;
    let x_correction = {
        let vector = vector.lock().unwrap();
        let matrix = matrix.lock().unwrap();
        if matrix.width() == vector.size() + 1 {
            -1
        } else {
            0
        }
    };
    let global = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let params = vec![
        ("src_vector", ParameterValue::Array(vector.clone())),
        ("src_matrix", ParameterValue::Array(matrix.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("x_correction", ParameterValue::Int(x_correction)),
    ];

    execute(device, kernel, &params, global, [0, 0, 0], constants)?;
    Ok(dst)
}

/// Compute the mean value among each label's touching neighbors.
pub fn mean_of_touching_neighbors(
    device: &DeviceArc,
    vector: &ArrayPtr,
    matrix: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    touching_neighbors_op(
        device,
        vector,
        matrix,
        dst,
        (
            "mean_touching_neighbors",
            include_str!("../../kernels/mean_touching_neighbors.cl"),
        ),
        &[],
    )
}

/// Compute the median value among each label's touching neighbors.
pub fn median_of_touching_neighbors(
    device: &DeviceArc,
    vector: &ArrayPtr,
    matrix: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let constants = [("MAX_ARRAY_SIZE", ConstantValue::Int(256))];
    touching_neighbors_op(
        device,
        vector,
        matrix,
        dst,
        (
            "median_touching_neighbors",
            include_str!("../../kernels/median_touching_neighbors.cl"),
        ),
        &constants,
    )
}

/// Compute the minimum value among each label's touching neighbors.
pub fn minimum_of_touching_neighbors(
    device: &DeviceArc,
    vector: &ArrayPtr,
    matrix: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    touching_neighbors_op(
        device,
        vector,
        matrix,
        dst,
        (
            "minimum_touching_neighbors",
            include_str!("../../kernels/minimum_touching_neighbors.cl"),
        ),
        &[],
    )
}

/// Compute the maximum value among each label's touching neighbors.
pub fn maximum_of_touching_neighbors(
    device: &DeviceArc,
    vector: &ArrayPtr,
    matrix: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    touching_neighbors_op(
        device,
        vector,
        matrix,
        dst,
        (
            "maximum_touching_neighbors",
            include_str!("../../kernels/maximum_touching_neighbors.cl"),
        ),
        &[],
    )
}

/// Compute the standard deviation among each label's touching neighbors.
pub fn standard_deviation_of_touching_neighbors(
    device: &DeviceArc,
    vector: &ArrayPtr,
    matrix: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    touching_neighbors_op(
        device,
        vector,
        matrix,
        dst,
        (
            "standard_deviation_touching_neighbors",
            include_str!("../../kernels/standard_deviation_touching_neighbors.cl"),
        ),
        &[],
    )
}

/// Compute the mode value among each label's touching neighbors.
pub fn mode_of_touching_neighbors(
    device: &DeviceArc,
    vector: &ArrayPtr,
    matrix: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    touching_neighbors_op(
        device,
        vector,
        matrix,
        dst,
        (
            "mode_touching_neighbors",
            include_str!("../../kernels/mode_touching_neighbors.cl"),
        ),
        &[],
    )
}
