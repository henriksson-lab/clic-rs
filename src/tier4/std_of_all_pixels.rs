use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier3;
use crate::types::DType;

pub fn standard_deviation_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let mean = tier3::mean_of_all_pixels(device, src)?;
    let diff_dst = tier0::create_like(src, None, DType::Float, device)?;
    let diff = tier1::subtract_scalar_from_image(device, src, Some(diff_dst), mean)?;
    let squared = tier1::power(device, &diff, None, 2.0)?;
    Ok(tier3::mean_of_all_pixels(device, &squared)?.sqrt())
}
