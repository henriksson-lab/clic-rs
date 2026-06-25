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
    let dst = tier0::create_vector(src, dst, num_bins as usize, INDEX, device)?;
    let (height, mtype, src_device) = {
        let src = src.lock().unwrap();
        (src.height(), src.mtype(), src.device().clone())
    };
    let number_of_partial_histograms = height;
    let partial_hist = Array::create(
        num_bins as usize,
        1,
        number_of_partial_histograms,
        3,
        INDEX,
        mtype,
        &src_device,
    )?;

    let mut minimum_intensity = minimum_intensity;
    let mut maximum_intensity = maximum_intensity;
    if minimum_intensity.is_nan() || maximum_intensity.is_nan() {
        minimum_intensity = tier2::minimum_of_all_pixels(device, src)?;
        maximum_intensity = tier2::maximum_of_all_pixels(device, src)?;
    }

    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(partial_hist.clone())),
        ("minimum", ParameterValue::Float(minimum_intensity)),
        ("maximum", ParameterValue::Float(maximum_intensity)),
        ("step_size_x", ParameterValue::Int(1)),
        ("step_size_y", ParameterValue::Int(1)),
        ("step_size_z", ParameterValue::Int(1)),
    ];
    let constants = vec![("NUMBER_OF_HISTOGRAM_BINS", ConstantValue::Int(num_bins))];
    let range = [number_of_partial_histograms, 1, 1];
    let local_range = [0, 0, 0];
    execute(
        device,
        ("histogram", include_str!("../../kernels/histogram.cl")),
        &params,
        range,
        local_range,
        &constants,
    )?;
    tier1::sum_z_projection(device, &partial_hist, Some(dst))
}
