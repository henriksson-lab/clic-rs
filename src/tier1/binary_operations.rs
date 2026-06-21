use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, KernelInfo, ParameterValue};
use crate::tier0;
use crate::types::{DType, BINARY};

use super::copy::copy;

fn binary_input(device: &DeviceArc, src: &ArrayPtr, op_name: &str) -> Result<ArrayPtr> {
    if src.lock().unwrap().dtype() == BINARY {
        return Ok(src.clone());
    }

    eprintln!(
        "Warning: Source image of {} expected to be binary, {:?} given.",
        op_name,
        src.lock().unwrap().dtype()
    );
    let tmp = tier0::create_like(src, None, BINARY, device)?;
    copy(device, src, Some(tmp.clone()))?;
    Ok(tmp)
}

fn execute_binary_op(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    op_name: &str,
    kernel_2d: KernelInfo,
    kernel_3d: KernelInfo,
) -> Result<ArrayPtr> {
    let src_binary = binary_input(device, src, op_name)?;
    let dst = tier0::create_like(src, dst, BINARY, device)?;
    let global = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let kernel = if src_binary.lock().unwrap().depth() > 1 {
        kernel_3d
    } else {
        kernel_2d
    };
    let params = vec![
        ("src", ParameterValue::Array(src_binary)),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(device, kernel, &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

pub fn binary_supinf(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    execute_binary_op(
        device,
        src,
        dst,
        "binary_supinf",
        (
            "superior_inferior",
            include_str!("../../kernels/superior_inferior_2d.cl"),
        ),
        (
            "superior_inferior",
            include_str!("../../kernels/superior_inferior_3d.cl"),
        ),
    )
}

pub fn binary_infsup(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    execute_binary_op(
        device,
        src,
        dst,
        "binary_infsup",
        (
            "inferior_superior",
            include_str!("../../kernels/inferior_superior_2d.cl"),
        ),
        (
            "inferior_superior",
            include_str!("../../kernels/inferior_superior_3d.cl"),
        ),
    )
}

pub fn binary_edge_detection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Uint8, device)?;
    let global = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        (
            "binary_edge_detection",
            include_str!("../../kernels/binary_edge_detection.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
