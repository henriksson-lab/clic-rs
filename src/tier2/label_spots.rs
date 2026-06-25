use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier1;
use crate::types::{DType, MType, LABEL};

/// Transform a binary spots image into a same-sized label image.
///
/// Mirrors CLIc's `label_spots_func`.
pub fn label_spots(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    dst.lock().unwrap().fill(0.0)?;

    let spot_count_in_x = tier1::sum_x_projection(device, src, None)?;
    let spot_count_in_xy = tier1::sum_y_projection(device, &spot_count_in_x, None)?;
    let range = {
        let dst = dst.lock().unwrap();
        [1, dst.height(), dst.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("countX", ParameterValue::Array(spot_count_in_x)),
        ("countXY", ParameterValue::Array(spot_count_in_xy)),
    ];
    execute(
        device,
        (
            "label_spots_in_x",
            include_str!("../../kernels/label_spots_in_x.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

/// Convert a pointlist image into labelled spots in a label image.
///
/// Mirrors CLIc's `pointlist_to_labelled_spots_func`.
pub fn pointlist_to_labelled_spots(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = if dst.is_none() {
        // Determine the number of dimensions of the pointlist n x ndims x 1
        let ndims = {
            let src = src.lock().unwrap();
            src.height()
        };
        let mut max_value = vec![0.0_f32; ndims];

        // Determine the maximum in each dimension, it should return a 1 x ndims x 1 array
        let max = tier1::maximum_x_projection(device, src, None)?;
        max.lock()
            .unwrap()
            .read_to_region(&mut max_value, [1, ndims, 1], [0, 0, 0])?;

        let width = if ndims > 0 {
            max_value[0] as usize + 1
        } else {
            1
        };
        let height = if ndims > 1 {
            max_value[1] as usize + 1
        } else {
            1
        };
        let depth = if ndims > 2 {
            max_value[2] as usize + 1
        } else {
            1
        };

        // Create destination with given dimensions
        Array::create(width, height, depth, ndims, LABEL, MType::Buffer, device)?
    } else {
        dst.unwrap()
    };
    dst.lock().unwrap().fill(0.0)?;

    let (width, height) = {
        let src = src.lock().unwrap();
        (src.width(), src.height())
    };
    let temp1 = Array::create(width, height + 1, 1, 2, DType::Float, MType::Buffer, device)?;
    let temp2 = Array::create(width, height + 1, 1, 2, DType::Float, MType::Buffer, device)?;

    tier1::set_ramp_x(device, &temp1)?;
    let temp2 = tier1::add_image_and_scalar(device, &temp1, Some(temp2), 1.0)?;
    src.lock()
        .unwrap()
        .copy_to_region(&temp2, [width, height, 1], [0, 0, 0], [0, 0, 0])?;

    tier1::write_values_to_positions(device, &temp2, Some(dst.clone()))?;
    Ok(dst)
}
