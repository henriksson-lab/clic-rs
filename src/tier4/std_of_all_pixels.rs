use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{evaluate, ParameterValue};
use crate::tier3;
use crate::types::DType;

pub fn standard_deviation_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let mean = tier3::mean_of_all_pixels(device, src)?;
    let diff = {
        let src = src.lock().unwrap();
        Array::create(
            src.width(),
            src.height(),
            src.depth(),
            src.dim(),
            DType::Float,
            src.mtype(),
            device,
        )?
    };
    evaluate(
        device,
        "pow(src - mean, 2)",
        &[
            ParameterValue::Array(src.clone()),
            ParameterValue::Float(mean),
        ],
        &diff,
    )?;
    let variance = tier3::mean_of_all_pixels(device, &diff)?;
    Ok(variance.sqrt())
}
