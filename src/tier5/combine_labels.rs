use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::tier4;
use crate::types::LABEL;

/// Combine two label images, with labels from `src1` overwriting `src0`.
///
/// Mirrors CLIc's `combine_labels_func`.
pub fn combine_labels(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src0, dst, LABEL, device)?;
    let max_label = tier2::maximum_of_all_pixels(device, src0)?;
    let mut temp1 = tier1::add_image_and_scalar(device, src1, None, max_label)?;
    let temp2 = tier1::greater_constant(device, src1, None, 0.0)?;
    temp1 = tier1::mask(device, &temp1, &temp2, None)?;
    let temp2 = tier1::maximum_images(device, src0, &temp1, None)?;
    tier4::relabel_sequential(device, &temp2, Some(dst), 4096)
}
