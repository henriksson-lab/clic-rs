use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Read image intensities at positions specified as point-list columns.
///
/// Mirrors CLIc's `read_values_from_positions_func`.
pub fn read_values_from_positions(
    device: &DeviceArc,
    src: &ArrayPtr,
    list: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let length = list.lock().unwrap().width();
    let dst = tier0::create_vector(src, dst, length, DType::Unknown, device)?;
    let kernel = (
        "read_values_from_positions",
        include_str!("../../kernels/read_values_from_positions.cl"),
    );
    let params = vec![
        ("src0", ParameterValue::Array(src.clone())),
        ("src1", ParameterValue::Array(list.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
