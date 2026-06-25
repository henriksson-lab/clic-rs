use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Paste `src` into `dst` at the given destination origin.
pub fn paste(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    destination_x: i32,
    destination_y: i32,
    destination_z: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let kernel = ("paste", include_str!("../../kernels/paste.cl"));
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("scalar0", ParameterValue::Int(destination_x)),
        ("scalar1", ParameterValue::Int(destination_y)),
        ("scalar2", ParameterValue::Int(destination_z)),
    ];
    let range = {
        let src = src.lock().unwrap();
        [src.width(), src.height(), src.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
