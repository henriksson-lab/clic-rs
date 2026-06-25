use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier1;
use crate::types::BINARY;

pub fn detect_minima(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, BINARY, device)?;
    let temp = tier1::mean_filter(
        device,
        src,
        None,
        radius_x,
        radius_y,
        radius_z,
        connectivity,
    )?;
    let kernel = (
        "detect_minima",
        include_str!("../../kernels/detect_minima.cl"),
    );
    let params = vec![
        ("src", ParameterValue::Array(temp)),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

pub fn detect_minima_box(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    radius_z: f32,
) -> Result<ArrayPtr> {
    detect_minima(device, src, dst, radius_x, radius_y, radius_z, "box")
}
