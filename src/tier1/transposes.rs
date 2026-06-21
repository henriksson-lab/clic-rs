use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::utils::shape_to_dimension;

fn create_transpose_dst(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mode: &str,
) -> Result<ArrayPtr> {
    if let Some(dst) = dst {
        return Ok(dst);
    }

    let src = src.lock().unwrap();
    let (width, height, depth, dim) = match mode {
        "XY" => {
            let width = src.height();
            let height = src.width();
            let depth = src.depth();
            (
                width,
                height,
                depth,
                shape_to_dimension(width, height, depth),
            )
        }
        "XZ" => {
            let width = src.depth();
            let height = src.height();
            let depth = src.width();
            (
                width,
                height,
                depth,
                shape_to_dimension(width, height, depth),
            )
        }
        "YZ" => (src.width(), src.depth(), src.height(), 3),
        _ => unreachable!("unsupported transpose mode"),
    };
    Array::create(width, height, depth, dim, src.dtype(), src.mtype(), device)
}

fn transpose(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mode: &str,
) -> Result<ArrayPtr> {
    let dst = create_transpose_dst(device, src, dst, mode)?;
    let global = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![("TRANSPOSE_MODE", ConstantValue::Str(mode.to_string()))];
    execute(
        device,
        ("transpose", include_str!("../../kernels/transpose.cl")),
        &params,
        global,
        [1, 1, 1],
        &constants,
    )?;
    Ok(dst)
}

pub fn transpose_xy(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    transpose(device, src, dst, "XY")
}

pub fn transpose_xz(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    transpose(device, src, dst, "XZ")
}

pub fn transpose_yz(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    transpose(device, src, dst, "YZ")
}
