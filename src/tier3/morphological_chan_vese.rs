use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{evaluate, ParameterValue};
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::types::{DType, MType, BINARY};

fn create_checkerboard_init(_src: &ArrayPtr, dst: &ArrayPtr, square_size: usize) -> Result<()> {
    let (width, height, depth) = {
        let dst = dst.lock().unwrap();
        (dst.width(), dst.height(), dst.depth())
    };
    let mut checkerboard = vec![0_u8; width * height * depth];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                // Calculate the index for the 1D vector
                let index = z * width * height + y * width + x;
                // Determine the checkerboard pattern value
                checkerboard[index] =
                    (((x / square_size) + (y / square_size) + (z / square_size)) % 2) as u8;
            }
        }
    }
    dst.lock().unwrap().write_from(&checkerboard)
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
    let gradient_along_axis = Array::create_from_array(gradient_magnitude)?;
    let absolute_gradient_along_axis = Array::create_from_array(gradient_magnitude)?;
    gradient_magnitude.lock().unwrap().fill(0.0)?;
    let dim = contour.lock().unwrap().dimension();
    for d in 0..dim {
        match d {
            0 => {
                tier1::gradient_x(device, contour, Some(gradient_along_axis.clone()))?;
            }
            1 => {
                tier1::gradient_y(device, contour, Some(gradient_along_axis.clone()))?;
            }
            2 => {
                tier1::gradient_z(device, contour, Some(gradient_along_axis.clone()))?;
            }
            _ => {}
        };
        tier1::absolute(
            device,
            &gradient_along_axis,
            Some(absolute_gradient_along_axis.clone()),
        )?;
        tier1::add_images_weighted(
            device,
            &absolute_gradient_along_axis,
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
    // magnitude * (lambda1 * (image - c1) ** 2 - lambda2 * (image - c0) ** 2)
    evaluate(
        device,
        "e * (l1 * pow(a + c1, 2.0f) - l2 * pow(a + c0, 2.0f))",
        &[
            ParameterValue::Array(evolution.clone()),
            ParameterValue::Float(lambda1),
            ParameterValue::Array(image.clone()),
            ParameterValue::Float(c1),
            ParameterValue::Float(lambda2),
            ParameterValue::Float(c0),
        ],
        evolution,
    )
}

fn apply_contour_evolution(
    device: &DeviceArc,
    evolution: &ArrayPtr,
    contour: &ArrayPtr,
) -> Result<()> {
    // auto evolution_pos = tier1::greater_constant_func(device, evolution, nullptr, 0);
    // auto evolution_neg = tier1::smaller_constant_func(device, evolution, nullptr, 0);
    // auto evolution_or = tier1::binary_or_func(device, evolution_pos, evolution_neg, nullptr);
    // auto mask = tier1::binary_not_func(device, evolution_or, nullptr);
    // auto masked_evolution = tier1::mask_func(device, contour, mask, nullptr);
    // tier1::add_images_weighted_func(device, masked_evolution, evolution_neg, contour, 1, 1);

    evaluate(
        device,
        "b * (a == 0.0f) + (a < 0.0f)",
        &[
            ParameterValue::Array(contour.clone()),
            ParameterValue::Array(evolution.clone()),
        ],
        contour,
    )
}

fn smooth_contour(device: &DeviceArc, dst: &ArrayPtr, iteration: i32) -> Result<()> {
    if iteration == 0 {
        return Ok(());
    }
    let temp = tier0::create_like(dst, None, DType::Unknown, device)?;
    for i in 0..iteration {
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
    // WARINING: dst MUST be binary
    let dst = match dst {
        Some(dst) => dst,
        None => {
            // dst is the initialisation contour
            // if not provided, use a checkerboard pattern as initialisation
            let dst = tier0::create_like(src, None, BINARY, device)?;
            create_checkerboard_init(src, &dst, 5)?;
            dst
        }
    };
    // enforce contour (dst) to be binary
    tier1::greater_constant(device, &dst, Some(dst.clone()), 0.0)?;

    let mut c0: f32;
    let mut c1: f32;
    let outside_contour = Array::create_from_array(&dst)?;
    let (width, height, depth, dim) = {
        let dst = dst.lock().unwrap();
        (dst.width(), dst.height(), dst.depth(), dst.dimension())
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

    let mut ite = 0;
    while ite < num_iter {
        // compute of inside contour score
        c1 = compute_contour_score(device, src, &dst)?;
        // compute of outside contour score (on inverted dst)
        tier1::binary_not(device, &dst, Some(outside_contour.clone()))?;
        c0 = compute_contour_score(device, src, &outside_contour)?;

        // compute gradient magnitude into temp_3
        compute_gradient_magnitude(device, &dst, &gradient_magnitude)?;

        // compute contour evolution according to gradient and score on contour
        compute_contour_evolution(device, src, &gradient_magnitude, c0, c1, lambda1, lambda2)?;

        // apply contour update on contour image
        apply_contour_evolution(device, &gradient_magnitude, &dst)?;

        // smooth contour
        smooth_contour(device, &dst, smoothing)?;

        ite += 1;
    }

    Ok(dst)
}
