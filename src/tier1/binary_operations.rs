use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::BINARY;

use super::copy::copy;

pub fn binary_supinf(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let src_binary = if src.lock().unwrap().dtype() != BINARY {
        eprintln!(
            "Warning: Source image of binary_supinf expected to be binary, {:?} given.",
            src.lock().unwrap().dtype()
        );
        let tmp = tier0::create_like(src, None, BINARY, device)?;
        copy(device, src, Some(tmp.clone()))?;
        tmp
    } else {
        src.clone()
    };
    let dst = tier0::create_like(src, dst, BINARY, device)?;
    let kernel = if src_binary.lock().unwrap().depth() > 1 {
        (
            "superior_inferior",
            include_str!("../../kernels/superior_inferior_3d.cl"),
        )
    } else {
        (
            "superior_inferior",
            include_str!("../../kernels/superior_inferior_2d.cl"),
        )
    };
    let params = vec![
        ("src", ParameterValue::Array(src_binary)),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

pub fn binary_infsup(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let src_binary = if src.lock().unwrap().dtype() != BINARY {
        eprintln!(
            "Warning: Source image of binary_infsup expected to be binary, {:?} given.",
            src.lock().unwrap().dtype()
        );
        let tmp = tier0::create_like(src, None, BINARY, device)?;
        copy(device, src, Some(tmp.clone()))?;
        tmp
    } else {
        src.clone()
    };
    let dst = tier0::create_like(src, dst, BINARY, device)?;
    let kernel = if src_binary.lock().unwrap().depth() > 1 {
        (
            "inferior_superior",
            include_str!("../../kernels/inferior_superior_3d.cl"),
        )
    } else {
        (
            "inferior_superior",
            include_str!("../../kernels/inferior_superior_2d.cl"),
        )
    };
    let params = vec![
        ("src", ParameterValue::Array(src_binary)),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

pub fn binary_edge_detection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, BINARY, device)?;
    let kernel = (
        "binary_edge_detection",
        include_str!("../../kernels/binary_edge_detection.cl"),
    );
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
