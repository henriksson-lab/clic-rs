use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier1;
use crate::types::{DType, MType};

pub fn hessian_gaussian_eigenvalues(
    device: &DeviceArc,
    src: &ArrayPtr,
    sigma: f32,
) -> Result<Vec<ArrayPtr>> {
    let (src_width, src_height, src_depth, src_dim) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.depth(), src.dim())
    };

    let small_eigenvalue = tier0::create_like(src, None, DType::Float, device)?;
    let large_eigenvalue = tier0::create_like(src, None, DType::Float, device)?;
    let middle_eigenvalue = if src_depth > 1 {
        tier0::create_like(src, None, DType::Float, device)?
    } else {
        tier0::create_one_like(src, None, DType::Float, device)?
    };

    let sigma = sigma * (1.0 / 2.0_f32.sqrt());
    let radius = (8.0 * sigma + 0.5) as usize;
    let kernel_size = 2 * radius + 1;
    let dirac_width = if src_width > 1 { kernel_size } else { 1 };
    let dirac_height = if src_height > 1 { kernel_size } else { 1 };
    let dirac_depth = if src_depth > 1 { kernel_size } else { 1 };
    let mut dirac_data = vec![0.0_f32; dirac_width * dirac_height * dirac_depth];
    let center = (dirac_width / 2)
        + (dirac_height / 2) * dirac_width
        + (dirac_depth / 2) * dirac_width * dirac_height;
    dirac_data[center] = 1.0;
    let dirac = Array::create_with_data(
        dirac_width,
        dirac_height,
        dirac_depth,
        src_dim,
        MType::Buffer,
        &dirac_data,
        device,
    )?;

    let gx = tier1::gaussian_derivative(device, &dirac, None, sigma, sigma, sigma, 1, 0, 0)?;
    let gxx = tier1::gaussian_derivative(device, &gx, None, sigma, sigma, sigma, 1, 0, 0)?;
    let gxy = tier1::gaussian_derivative(device, &gx, None, sigma, sigma, sigma, 0, 1, 0)?;

    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("g_xx", ParameterValue::Array(gxx)),
        ("g_xy", ParameterValue::Array(gxy)),
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
            "hessian_gaussian_eigenvalues",
            include_str!("../../kernels/hessian_gaussian_eigenvalues.cl"),
        ),
        &params,
        [src_width, src_height, src_depth],
        [0, 0, 0],
        &[],
    )?;

    if src_depth == 1 {
        Ok(vec![large_eigenvalue, small_eigenvalue])
    } else {
        Ok(vec![large_eigenvalue, middle_eigenvalue, small_eigenvalue])
    }
}
