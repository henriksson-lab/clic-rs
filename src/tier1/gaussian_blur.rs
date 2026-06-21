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
