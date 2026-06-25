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
    small_eigenvalue: Option<ArrayPtr>,
    middle_eigenvalue: Option<ArrayPtr>,
    large_eigenvalue: Option<ArrayPtr>,
    sigma: f32,
) -> Result<Vec<ArrayPtr>> {
    // TODO: check when src is 1D
    let small_eigenvalue = tier0::create_like(src, small_eigenvalue, DType::Float, device)?;
    let large_eigenvalue = tier0::create_like(src, large_eigenvalue, DType::Float, device)?;
    let src_depth = src.lock().unwrap().depth();
    let middle_eigenvalue = if src_depth > 1 {
        tier0::create_like(src, middle_eigenvalue, DType::Float, device)?
    } else {
        // no middle eigenvalue for 2D images, we replace the image by a scalar to save memory
        tier0::create_one(src, middle_eigenvalue, DType::Float, device)?
    };

    // scale sigma
    // (https://github.com/scikit-image/scikit-image/blob/be7ff3442864f8c44236c4cd50c04039a85b3ea8/skimage/feature/corner.py#L196)
    let one = 1.0_f32; // std::sqrt(2.0F);
    let truncate = 8;
    let sq1_2 = 1.0_f32 / 2.0_f32.sqrt();
    let mut sigma = sigma;
    sigma *= sq1_2;

    let radius = (truncate as f32 * sigma + 0.5) as usize;
    let kernel_size = 2 * radius + 1;
    let (src_width, src_height, src_depth, src_dim) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.depth(), src.dim())
    };
    let dirac = Array::create(
        if src_width > 1 { kernel_size } else { 1 },
        if src_height > 1 { kernel_size } else { 1 },
        if src_depth > 1 { kernel_size } else { 1 },
        src_dim,
        DType::Float,
        MType::Buffer,
        device,
    )?;
    dirac.lock().unwrap().fill(0.0)?;
    let (dirac_width, dirac_height, dirac_depth) = {
        let dirac = dirac.lock().unwrap();
        (dirac.width(), dirac.height(), dirac.depth())
    };
    dirac.lock().unwrap().write_from_region(
        &[one],
        [1, 1, 1],
        [dirac_width / 2, dirac_height / 2, dirac_depth / 2],
    )?;

    // compute the Gaussian first derivatives along x and blur along y to get Gx
    let g_x = tier1::gaussian_derivative(device, &dirac, None, sigma, sigma, sigma, 1, 0, 0)?;
    // compute the Gaussian second derivatives along x and blur along y to get Gxx
    // compute the Gaussian second derivatives along y and blur along x to get Gxy
    let g_xx = tier1::gaussian_derivative(device, &g_x, None, sigma, sigma, sigma, 1, 0, 0)?;
    let g_xy = tier1::gaussian_derivative(device, &g_x, None, sigma, sigma, sigma, 0, 1, 0)?;

    let kernel = (
        "hessian_gaussian_eigenvalues",
        include_str!("../../kernels/hessian_gaussian_eigenvalues.cl"),
    );
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("g_xx", ParameterValue::Array(g_xx)),
        ("g_xy", ParameterValue::Array(g_xy)),
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
    let range = [src_width, src_height, src_depth];
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;

    if src_depth == 1 {
        Ok(vec![large_eigenvalue, small_eigenvalue])
    } else {
        Ok(vec![large_eigenvalue, middle_eigenvalue, small_eigenvalue])
    }
}
