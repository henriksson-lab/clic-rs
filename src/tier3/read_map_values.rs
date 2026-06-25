use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier2;
use crate::types::DType;

/// Read values from a parametric map using its corresponding labels.
///
/// Returns a float vector with one element per label index, including the
/// background entry at index 0.
pub fn read_map_values(
    device: &DeviceArc,
    map: &ArrayPtr,
    label: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let max_label = tier2::maximum_of_all_pixels(device, label)? + 1.0;
    let dst = tier0::create_vector(label, dst, max_label as usize, DType::Float, device)?;
    dst.lock().unwrap().fill(0.0)?;

    let params = vec![
        ("src0", ParameterValue::Array(label.clone())),
        ("src1", ParameterValue::Array(map.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let kernel = (
        "read_map_values",
        include_str!("../../kernels/read_map_values.cl"),
    );
    let range = {
        let label = label.lock().unwrap();
        [label.width(), label.height(), label.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Deprecated CLIc alias for [`read_map_values`].
pub fn read_intensities_from_map(
    device: &DeviceArc,
    label: &ArrayPtr,
    map: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    read_map_values(device, map, label, dst)
}
