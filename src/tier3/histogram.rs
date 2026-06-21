use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::types::INDEX;

pub fn histogram(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    num_bins: i32,
    minimum_intensity: f32,
    maximum_intensity: f32,
) -> Result<ArrayPtr> {
    let num_bins = num_bins.max(1) as usize;
    let dst = match dst {
        Some(dst) => dst,
        None => tier0::create_vector(num_bins, INDEX, device)?,
    };
    let (height, mtype) = {
        let src = src.lock().unwrap();
        (src.height(), src.mtype())
    };
    let partial_hist = Array::create(num_bins, 1, height, 3, INDEX, mtype, device)?;

    let (minimum_intensity, maximum_intensity) =
        if minimum_intensity.is_nan() || maximum_intensity.is_nan() {
            (
                tier2::minimum_of_all_pixels(device, src)?,
                tier2::maximum_of_all_pixels(device, src)?,
            )
        } else {
            (minimum_intensity, maximum_intensity)
        };

    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(partial_hist.clone())),
        ("minimum", ParameterValue::Float(minimum_intensity)),
        ("maximum", ParameterValue::Float(maximum_intensity)),
        ("step_size_x", ParameterValue::Int(1)),
        ("step_size_y", ParameterValue::Int(1)),
        ("step_size_z", ParameterValue::Int(1)),
    ];
    let constants = vec![(
        "NUMBER_OF_HISTOGRAM_BINS",
        ConstantValue::Int(num_bins as i32),
    )];
    execute(
        device,
        ("histogram", include_str!("../../kernels/histogram.cl")),
        &params,
        [height, 1, 1],
        [0, 0, 0],
        &constants,
    )?;
    tier1::sum_z_projection(device, &partial_hist, Some(dst))
}
