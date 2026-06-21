use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier6;
use crate::types::LABEL;

/// Apply morphological opening to a label image.
///
/// Mirrors CLIc's `opening_labels_func`.
pub fn opening_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    let temp = tier6::erode_labels(device, src, None, radius, false)?;
    tier6::dilate_labels(device, &temp, Some(dst), radius)
}
