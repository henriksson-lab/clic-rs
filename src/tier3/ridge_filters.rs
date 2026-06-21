use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier2;
use crate::types::DType;

const SATO_2D_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void sato_update_2d(
    IMAGE_small_TYPE small,
    IMAGE_dst_TYPE dst,
    const float sigma_squared
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const float a = (float) READ_IMAGE(small, sampler, POS_small_INSTANCE(x,y,z,0)).x;
    const float out = (float) READ_IMAGE(dst, sampler, POS_dst_INSTANCE(x,y,z,0)).x;
    const float value = fmax(fabs(fmin(a, 0.0f)) * sigma_squared, out);
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}
"#;

const SATO_3D_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void sato_update_3d(
    IMAGE_middle_TYPE middle,
    IMAGE_small_TYPE small,
    IMAGE_dst_TYPE dst,
    const float sigma_squared
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const float a = (float) READ_IMAGE(middle, sampler, POS_middle_INSTANCE(x,y,z,0)).x;
    const float b = (float) READ_IMAGE(small, sampler, POS_small_INSTANCE(x,y,z,0)).x;
    const float out = (float) READ_IMAGE(dst, sampler, POS_dst_INSTANCE(x,y,z,0)).x;
    const float value = fmax(sqrt(fmin(a, 0.0f) * fmin(b, 0.0f)) * sigma_squared, out);
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}
"#;

fn global_from(arr: &ArrayPtr) -> [usize; 3] {
    let arr = arr.lock().unwrap();
    [arr.width(), arr.height(), arr.depth()]
}

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
    let is_3d = src.lock().unwrap().depth() > 1;
    let global = global_from(src);

    let mut sigma = sigma_minimum;
    while sigma < sigma_maximum {
        let sigma_squared = sigma * sigma;
        let eigenvalues = tier2::hessian_gaussian_eigenvalues(device, src, sigma)?;
        let small = eigenvalues
            .last()
            .expect("hessian_gaussian_eigenvalues returns at least one eigenvalue")
            .clone();
        if is_3d {
            let middle = eigenvalues[1].clone();
            let params = vec![
                ("middle", ParameterValue::Array(middle)),
                ("small", ParameterValue::Array(small)),
                ("dst", ParameterValue::Array(dst.clone())),
                ("sigma_squared", ParameterValue::Float(sigma_squared)),
            ];
            execute(
                device,
                ("sato_update_3d", SATO_3D_SRC),
                &params,
                global,
                [0, 0, 0],
                &[],
            )?;
        } else {
            let params = vec![
                ("small", ParameterValue::Array(small)),
                ("dst", ParameterValue::Array(dst.clone())),
                ("sigma_squared", ParameterValue::Float(sigma_squared)),
            ];
            execute(
                device,
                ("sato_update_2d", SATO_2D_SRC),
                &params,
                global,
                [0, 0, 0],
                &[],
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
