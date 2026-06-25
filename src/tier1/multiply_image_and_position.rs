use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Multiply all pixel intensities by the coordinate in `dimension`.
///
/// Mirrors CLIc's `multiply_image_and_position_func`.
pub fn multiply_image_and_position(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    dimension: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;
    let kernel = (
        "multiply_image_and_position",
        include_str!("../../kernels/multiply_image_and_position.cl"),
    );
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index", ParameterValue::Int(dimension)),
    ];
    let range = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
