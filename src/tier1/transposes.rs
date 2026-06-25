use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};

pub fn transpose_xy(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = if let Some(dst) = dst {
        dst
    } else {
        let src = src.lock().unwrap();
        Array::create(
            src.height(),
            src.width(),
            src.depth(),
            crate::utils::shape_to_dimension(src.height(), src.width(), src.depth()),
            src.dtype(),
            src.mtype(),
            &src.device().clone(),
        )?
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let local = [1, 1, 1];
    let constants = vec![("TRANSPOSE_MODE", ConstantValue::Str("XY".to_string()))];
    execute(
        device,
        ("transpose", include_str!("../../kernels/transpose.cl")),
        &params,
        range,
        local,
        &constants,
    )?;
    Ok(dst)
}

pub fn transpose_xz(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = if let Some(dst) = dst {
        dst
    } else {
        let src = src.lock().unwrap();
        Array::create(
            src.depth(),
            src.height(),
            src.width(),
            crate::utils::shape_to_dimension(src.depth(), src.height(), src.width()),
            src.dtype(),
            src.mtype(),
            &src.device().clone(),
        )?
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let local = [1, 1, 1];
    let constants = vec![("TRANSPOSE_MODE", ConstantValue::Str("XZ".to_string()))];
    execute(
        device,
        ("transpose", include_str!("../../kernels/transpose.cl")),
        &params,
        range,
        local,
        &constants,
    )?;
    Ok(dst)
}

pub fn transpose_yz(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = if let Some(dst) = dst {
        dst
    } else {
        let src = src.lock().unwrap();
        Array::create(
            src.width(),
            src.depth(),
            src.height(),
            3,
            src.dtype(),
            src.mtype(),
            &src.device().clone(),
        )?
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let local = [1, 1, 1];
    let constants = vec![("TRANSPOSE_MODE", ConstantValue::Str("YZ".to_string()))];
    execute(
        device,
        ("transpose", include_str!("../../kernels/transpose.cl")),
        &params,
        range,
        local,
        &constants,
    )?;
    Ok(dst)
}
