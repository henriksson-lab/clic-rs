use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier5;
use crate::types::DType;

/// Map each label to its proximal-neighbor count.
///
/// Mirrors CLIc's `proximal_neighbor_count_map_func`.
pub fn proximal_neighbor_count_map(
    device: &DeviceArc,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
    min_distance: f32,
    max_distance: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(labels, dst, DType::Uint32, device)?;
    let min_distance = min_distance.max(0.0);
    let max_distance = if max_distance < 0.0 {
        f32::MAX
    } else {
        max_distance
    };
    let counts = tier5::proximal_neighbor_count(device, labels, None, min_distance, max_distance)?;
    tier1::replace_intensities(device, labels, &counts, Some(dst))
}
