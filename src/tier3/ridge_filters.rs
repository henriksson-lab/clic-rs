use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{evaluate, ParameterValue};
use crate::tier0;
use crate::tier2;
use crate::types::{DType, MType};

pub fn sato_filter(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    sigma_minimum: f32,
    sigma_maximum: f32,
    sigma_step: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    dst.lock().unwrap().fill(0.0)?;
    let (width, height, depth, dim) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.depth(), src.dim())
    };
    let is_3d = depth > 1;

    let small_eigenvalue = Array::create(
        width,
        height,
        depth,
        dim,
        DType::Float,
        MType::Buffer,
        device,
    )?;
    let middle_eigenvalue = if is_3d {
        Some(Array::create(
            width,
            height,
            depth,
            dim,
            DType::Float,
            MType::Buffer,
            device,
        )?)
    } else {
        None
    };

    let mut sigma = sigma_minimum;
    while sigma < sigma_maximum {
        let sigma_squared = sigma * sigma;
        tier2::hessian_gaussian_eigenvalues(
            device,
            src,
            Some(small_eigenvalue.clone()),
            middle_eigenvalue.clone(),
            None,
            sigma,
        )?;
        if is_3d {
            evaluate(
                device,
                "fmax(sqrt(fmin(a, 0.0f) * fmin(b, 0.0f)) * s, out)",
                &[
                    ParameterValue::Array(middle_eigenvalue.as_ref().unwrap().clone()),
                    ParameterValue::Array(small_eigenvalue.clone()),
                    ParameterValue::Float(sigma_squared),
                    ParameterValue::Array(dst.clone()),
                ],
                &dst,
            )?;
        } else {
            evaluate(
                device,
                "fmax(fabs(fmin(a, 0.0f)) * s, out)",
                &[
                    ParameterValue::Array(small_eigenvalue.clone()),
                    ParameterValue::Float(sigma_squared),
                    ParameterValue::Array(dst.clone()),
                ],
                &dst,
            )?;
        }
        sigma += sigma_step;
    }

    Ok(dst)
}

pub fn tubeness(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    sigma: f32,
) -> Result<ArrayPtr> {
    sato_filter(device, src, dst, sigma, sigma + 0.1, 0.1)
}
