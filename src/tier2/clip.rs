use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

use super::minmax_of_all_pixels::{maximum_of_all_pixels, minimum_of_all_pixels};

const CLIP_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__kernel void clip_kernel(IMAGE_src_TYPE src, IMAGE_dst_TYPE dst, float lo, float hi) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    float v = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,z,0), CONVERT_dst_PIXEL_TYPE(clamp(v, lo, hi)));
}
"#;

/// Clamp pixel values to [min_intensity, max_intensity].
pub fn clip(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    min_intensity: f32,
    max_intensity: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let min_intensity = if min_intensity.is_nan() {
        minimum_of_all_pixels(device, src)?
    } else {
        min_intensity
    };
    let max_intensity = if max_intensity.is_nan() {
        maximum_of_all_pixels(device, src)?
    } else {
        max_intensity
    };
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("lo", ParameterValue::Float(min_intensity)),
        ("hi", ParameterValue::Float(max_intensity)),
    ];
    execute(
        device,
        ("clip_kernel", CLIP_SRC),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
