use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Compute Hessian eigenvalues.
///
/// Mirrors CLIc's `hessian_eigenvalues_func`. For 2D inputs this returns
/// `[large, small]`; for 3D inputs this returns `[large, middle, small]`.
pub fn hessian_eigenvalues(device: &DeviceArc, src: &ArrayPtr) -> Result<Vec<ArrayPtr>> {
    let depth = {
        let l = src.lock().unwrap();
        l.depth()
    };

    let small_eigenvalue = tier0::create_like(src, None, DType::Float, device)?;
    let large_eigenvalue = tier0::create_like(src, None, DType::Float, device)?;
    let middle_eigenvalue = if depth > 1 {
        tier0::create_like(src, None, DType::Float, device)?
    } else {
        tier0::create_one_like(src, None, DType::Float, device)?
    };

    let global = {
        let l = src.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        (
            "small_eigenvalue",
            ParameterValue::Array(small_eigenvalue.clone()),
        ),
        (
            "middle_eigenvalue",
            ParameterValue::Array(middle_eigenvalue.clone()),
        ),
        (
            "large_eigenvalue",
            ParameterValue::Array(large_eigenvalue.clone()),
        ),
    ];
    execute(
        device,
        (
            "hessian_eigenvalues",
            include_str!("../../kernels/hessian_eigenvalues.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;

    if depth == 1 {
        Ok(vec![large_eigenvalue, small_eigenvalue])
    } else {
        Ok(vec![large_eigenvalue, middle_eigenvalue, small_eigenvalue])
    }
}
