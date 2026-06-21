use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::tier7;
use crate::types::LABEL;

/// Smooth labels by opening, Voronoi gap filling, and background masking.
///
/// Mirrors CLIc's `smooth_labels_func`.
pub fn smooth_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    if radius < 1 {
        return tier1::copy(device, src, Some(dst));
    }
    let binary = tier1::greater_constant(device, src, None, 0.0)?;
    let opened = tier7::opening_labels(device, src, None, radius)?;
    let extended = tier2::extend_labeling_via_voronoi(device, &opened, None)?;
    tier1::multiply_images(device, &binary, &extended, Some(dst))
}
