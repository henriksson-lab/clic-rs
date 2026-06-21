use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::types::{DType, MType, BINARY};

const CONTOUR_EVOLUTION_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void contour_evolution(
    IMAGE_evolution_TYPE evolution,
    IMAGE_image_TYPE image,
    IMAGE_dst_TYPE dst,
    const float c0,
    const float c1,
    const float lambda1,
    const float lambda2
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const float e = (float) READ_IMAGE(evolution, sampler, POS_evolution_INSTANCE(x,y,z,0)).x;
    const float a = (float) READ_IMAGE(image, sampler, POS_image_INSTANCE(x,y,z,0)).x;
    const float value = e * (lambda1 * pow(a + c1, 2.0f) - lambda2 * pow(a + c0, 2.0f));
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(value));
}
"#;

const APPLY_CONTOUR_EVOLUTION_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void apply_contour_evolution(
    IMAGE_contour_TYPE contour,
    IMAGE_evolution_TYPE evolution
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const float b = (float) READ_IMAGE(contour, sampler, POS_contour_INSTANCE(x,y,z,0)).x;
    const float a = (float) READ_IMAGE(evolution, sampler, POS_evolution_INSTANCE(x,y,z,0)).x;
    const float value = b * (a == 0.0f) + (a < 0.0f);
    WRITE_IMAGE(contour, POS_contour_INSTANCE(x,y,z,0), CONVERT_contour_PIXEL_TYPE(value));
}
"#;

fn global_from(arr: &ArrayPtr) -> [usize; 3] {
    let arr = arr.lock().unwrap();
    [arr.width(), arr.height(), arr.depth()]
}

fn create_checkerboard_init(
    device: &DeviceArc,
    src: &ArrayPtr,
    square_size: usize,
) -> Result<ArrayPtr> {
    let (width, height, depth, dim, mtype) = {
        let src = src.lock().unwrap();
        (
            src.width(),
            src.height(),
            src.depth(),
            src.dim(),
            src.mtype(),
        )
    };
    let mut data = vec![0_u8; width * height * depth];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let index = z * width * height + y * width + x;
                data[index] =
                    (((x / square_size) + (y / square_size) + (z / square_size)) % 2) as u8;
            }
        }
    }
    Array::create_with_data(width, height, depth, dim, mtype, &data, device)
}

fn compute_contour_score(device: &DeviceArc, image: &ArrayPtr, contour: &ArrayPtr) -> Result<f32> {
    let masked = tier1::mask(device, image, contour, None)?;
    let sum_image_value = tier2::sum_of_all_pixels(device, &masked)?;
    let sum_contour_value = tier2::sum_of_all_pixels(device, contour)? + 1e-8;
    Ok(-sum_image_value / sum_contour_value)
}

fn compute_gradient_magnitude(
    device: &DeviceArc,
    contour: &ArrayPtr,
    gradient_magnitude: &ArrayPtr,
) -> Result<()> {
    gradient_magnitude.lock().unwrap().fill(0.0)?;
    let dim = contour.lock().unwrap().dimension();
    for axis in 0..dim {
        let gradient = match axis {
            0 => tier1::gradient_x(device, contour, None)?,
            1 => tier1::gradient_y(device, contour, None)?,
            _ => tier1::gradient_z(device, contour, None)?,
        };
        let absolute_gradient = tier1::absolute(device, &gradient, None)?;
        tier1::add_images_weighted(
            device,
            &absolute_gradient,
            gradient_magnitude,
            Some(gradient_magnitude.clone()),
            1.0,
            1.0,
        )?;
    }
    Ok(())
}

fn compute_contour_evolution(
    device: &DeviceArc,
    image: &ArrayPtr,
    evolution: &ArrayPtr,
    c0: f32,
    c1: f32,
    lambda1: f32,
    lambda2: f32,
) -> Result<()> {
    let params = vec![
        ("evolution", ParameterValue::Array(evolution.clone())),
        ("image", ParameterValue::Array(image.clone())),
        ("dst", ParameterValue::Array(evolution.clone())),
        ("c0", ParameterValue::Float(c0)),
        ("c1", ParameterValue::Float(c1)),
        ("lambda1", ParameterValue::Float(lambda1)),
        ("lambda2", ParameterValue::Float(lambda2)),
    ];
    execute(
        device,
        ("contour_evolution", CONTOUR_EVOLUTION_SRC),
        &params,
        global_from(evolution),
        [0, 0, 0],
        &[],
    )
}

fn apply_contour_evolution(
    device: &DeviceArc,
    evolution: &ArrayPtr,
    contour: &ArrayPtr,
) -> Result<()> {
    let params = vec![
        ("contour", ParameterValue::Array(contour.clone())),
        ("evolution", ParameterValue::Array(evolution.clone())),
    ];
    execute(
        device,
        ("apply_contour_evolution", APPLY_CONTOUR_EVOLUTION_SRC),
        &params,
        global_from(contour),
        [0, 0, 0],
        &[],
    )
}

fn smooth_contour(device: &DeviceArc, dst: &ArrayPtr, iterations: i32) -> Result<()> {
    if iterations == 0 {
        return Ok(());
    }
    let temp = tier0::create_like(dst, None, DType::Unknown, device)?;
    for i in 0..iterations {
        if i % 2 == 0 {
            tier1::binary_supinf(device, dst, Some(temp.clone()))?;
            tier1::binary_infsup(device, &temp, Some(dst.clone()))?;
        } else {
            tier1::binary_infsup(device, dst, Some(temp.clone()))?;
            tier1::binary_supinf(device, &temp, Some(dst.clone()))?;
        }
    }
    Ok(())
}

pub fn morphological_chan_vese(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    num_iter: i32,
    smoothing: i32,
    lambda1: f32,
    lambda2: f32,
) -> Result<ArrayPtr> {
    let dst = match dst {
        Some(dst) => dst,
        None => create_checkerboard_init(device, src, 5)?,
    };
    tier1::greater_constant(device, &dst, Some(dst.clone()), 0.0)?;

    let outside_contour = tier0::create_like(&dst, None, BINARY, device)?;
    let (width, height, depth, dim) = {
        let dst = dst.lock().unwrap();
        (dst.width(), dst.height(), dst.depth(), dst.dim())
    };
    let gradient_magnitude = Array::create(
        width,
        height,
        depth,
        dim,
        DType::Float,
        MType::Buffer,
        device,
    )?;

    for _ in 0..num_iter {
        let c1 = compute_contour_score(device, src, &dst)?;
        tier1::binary_not(device, &dst, Some(outside_contour.clone()))?;
        let c0 = compute_contour_score(device, src, &outside_contour)?;
        compute_gradient_magnitude(device, &dst, &gradient_magnitude)?;
        compute_contour_evolution(device, src, &gradient_magnitude, c0, c1, lambda1, lambda2)?;
        apply_contour_evolution(device, &gradient_magnitude, &dst)?;
        smooth_contour(device, &dst, smoothing)?;
    }

    Ok(dst)
}
