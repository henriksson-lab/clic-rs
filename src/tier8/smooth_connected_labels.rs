use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier6;
use crate::tier7;
use crate::types::LABEL;

/// Smooth connected labels by connected erosion followed by label dilation.
///
/// Mirrors CLIc's `smooth_connected_labels_func`.
pub fn smooth_connected_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    if radius < 1 {
        return tier1::copy(device, src, Some(dst));
    }
    let binary = tier7::erode_connected_labels(device, src, None, radius)?;
    tier6::dilate_labels(device, &binary, Some(dst), radius)
}
