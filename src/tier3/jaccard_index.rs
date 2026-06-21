use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;
use crate::tier2;

pub fn jaccard_index(device: &DeviceArc, src0: &ArrayPtr, src1: &ArrayPtr) -> Result<f32> {
    let intersection = tier1::binary_and(device, src0, src1, None)?;
    let union = tier1::binary_or(device, src0, src1, None)?;
    Ok(
        tier2::sum_of_all_pixels(device, &intersection)?
            / tier2::sum_of_all_pixels(device, &union)?,
    )
}
