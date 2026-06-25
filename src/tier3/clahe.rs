use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier2;
use crate::types::DType;

pub fn clahe(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    tile_size: i32,
    mut clip_limit: f32,
    mut minimum_intensity: f32,
    mut maximum_intensity: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    if minimum_intensity.is_nan() || maximum_intensity.is_nan() {
        minimum_intensity = tier2::minimum_of_all_pixels(device, src)?;
        maximum_intensity = tier2::maximum_of_all_pixels(device, src)?;
    }
    clip_limit *= 255.0;
    let kernel = ("clahe", include_str!("../../kernels/clahe.cl"));
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("tileSize", ParameterValue::Int(tile_size)),
        ("clipLimit", ParameterValue::Float(clip_limit)),
        ("minIntensity", ParameterValue::Float(minimum_intensity)),
        ("maxIntensity", ParameterValue::Float(maximum_intensity)),
    ];
    let range = {
        let src = src.lock().unwrap();
        [src.width(), src.height(), src.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
