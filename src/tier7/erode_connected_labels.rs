use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier4;
use crate::tier6;
use crate::types::LABEL;

/// Erode connected labels while preserving original label identity where possible.
///
/// Mirrors CLIc's `erode_connected_labels_func`.
pub fn erode_connected_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    if radius < 1 {
        return tier1::copy(device, src, Some(dst));
    }
    let temp = tier1::greater_constant(device, src, None, 0.0)?;
    let eroded = tier6::erode_labels(device, &temp, None, radius, false)?;
    let temp = tier1::multiply_images(device, src, &eroded, None)?;
    tier4::relabel_sequential(device, &temp, Some(dst), 4096)
}
