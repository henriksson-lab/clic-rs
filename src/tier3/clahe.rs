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
    clip_limit: f32,
    minimum_intensity: f32,
    maximum_intensity: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let (minimum_intensity, maximum_intensity) =
        if minimum_intensity.is_nan() || maximum_intensity.is_nan() {
            (
                tier2::minimum_of_all_pixels(device, src)?,
                tier2::maximum_of_all_pixels(device, src)?,
            )
        } else {
            (minimum_intensity, maximum_intensity)
        };
    let clip_limit = clip_limit * 255.0;
    let global = {
        let src = src.lock().unwrap();
        [src.width(), src.height(), src.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("tileSize", ParameterValue::Int(tile_size)),
        ("clipLimit", ParameterValue::Float(clip_limit)),
        ("minIntensity", ParameterValue::Float(minimum_intensity)),
        ("maxIntensity", ParameterValue::Float(maximum_intensity)),
    ];
    execute(
        device,
        ("clahe", include_str!("../../kernels/clahe.cl")),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
