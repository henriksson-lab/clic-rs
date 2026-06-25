use crate::array::{Array, ArrayPtr};
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

    let mut temp = src.clone();
    if temp.lock().unwrap().dtype() != DType::Float {
        temp = Array::create_from_array(&dst)?;
        copy(device, src, Some(temp.clone()))?;
    }

    let kernel = (
        "gaussian_blur_separable",
        include_str!("../../kernels/gaussian_blur_separable.cl"),
    );
    execute_separable(
        device,
        kernel,
        &temp,
        &dst,
        [sigma_x, sigma_y, sigma_z],
        [
            crate::utils::sigma2kernelsize(sigma_x),
            crate::utils::sigma2kernelsize(sigma_y),
            crate::utils::sigma2kernelsize(sigma_z),
        ],
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

    let mut temp = src.clone();
    if temp.lock().unwrap().dtype() != DType::Float {
        temp = Array::create_from_array(&dst)?;
        copy(device, src, Some(temp.clone()))?;
    }

    const TRUNCATE: f32 = 8.0;
    let sigmas = [sigma_x.max(0.0), sigma_y.max(0.0), sigma_z.max(0.0)];
    let radii = [
        (TRUNCATE * sigmas[0] + 0.5) as i32,
        (TRUNCATE * sigmas[1] + 0.5) as i32,
        (TRUNCATE * sigmas[2] + 0.5) as i32,
    ];
    let orders = [order_x.min(2), order_y.min(2), order_z.min(2)];

    let kernel = (
        "gaussian_derivative_separable",
        include_str!("../../kernels/gaussian_derivative_separable.cl"),
    );
    execute_separable(device, kernel, &temp, &dst, sigmas, radii, orders)?;
    Ok(dst)
}
