use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier3;
use crate::types::BINARY;

pub fn threshold_mean(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let mean_intensity = tier3::mean_of_all_pixels(device, src)?;
    let dst = tier0::create_like(src, dst, BINARY, device)?;
    tier1::greater_constant(device, src, Some(dst), mean_intensity)
}
