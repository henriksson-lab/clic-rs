use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::execute_separable;
use crate::tier0;
use crate::types::DType;

use super::copy::copy;

/// Gaussian blur with per-axis sigma values.
pub fn gaussian_blur(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    sigma_x: f32,
    sigma_y: f32,
    sigma_z: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;

    let src_float = if src.lock().unwrap().dtype() != DType::Float {
        let t = tier0::create_like(src, None, DType::Float, device)?;
        copy(device, src, Some(t.clone()))?
    } else {
        src.clone()
    };

    let sigma = [sigma_x, sigma_y, sigma_z];
    let radius = sigma.map(crate::utils::sigma2kernelsize);
    execute_separable(
        device,
        (
            "gaussian_blur_separable",
            include_str!("../../kernels/gaussian_blur_separable.cl"),
        ),
        &src_float,
        &dst,
        sigma,
        radius,
        [0, 0, 0],
    )?;
    Ok(dst)
}

/// Gaussian derivative with per-axis sigma and derivative order values.
pub fn gaussian_derivative(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    sigma_x: f32,
    sigma_y: f32,
    sigma_z: f32,
    order_x: i32,
    order_y: i32,
    order_z: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Float, device)?;

    let src_float = if src.lock().unwrap().dtype() != DType::Float {
        let t = tier0::create_like(src, None, DType::Float, device)?;
        copy(device, src, Some(t.clone()))?
    } else {
        src.clone()
    };

    const TRUNCATE: f32 = 8.0;
    let sigma = [sigma_x.max(0.0), sigma_y.max(0.0), sigma_z.max(0.0)];
    let radius = sigma.map(|s| (TRUNCATE * s + 0.5) as i32);
    let orders = [order_x.min(2), order_y.min(2), order_z.min(2)];

    execute_separable(
        device,
        (
            "gaussian_derivative_separable",
            include_str!("../../kernels/gaussian_derivative_separable.cl"),
        ),
        &src_float,
        &dst,
        sigma,
        radius,
        orders,
    )?;
    Ok(dst)
}
