use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::{tier1, tier2, tier4};

/// Count labels whose centroids are within a distance range of each other.
///
/// Returns a vector containing one count per label. Negative `max_distance`
/// means no upper bound, and negative `min_distance` is clamped to zero.
/// Mirrors CLIc's `proximal_neighbor_count_func`.
pub fn proximal_neighbor_count(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    min_distance: f32,
    max_distance: f32,
) -> Result<ArrayPtr> {
    let min_distance = min_distance.max(0.0);
    let max_distance = if max_distance < 0.0 {
        f32::MAX
    } else {
        max_distance
    };

    let pointlist = tier4::centroids_of_labels(device, src, None, false)?;
    let distance_matrix = tier1::generate_distance_matrix(device, &pointlist, &pointlist, None)?;
    let touch_matrix = tier2::generate_proximal_neighbors_matrix(
        device,
        &distance_matrix,
        None,
        min_distance,
        max_distance,
    )?;

    let dst = tier2::count_touching_neighbors(device, &touch_matrix, dst, false)?;
    tier1::set_column(device, &dst, 0, 0.0)?;

    Ok(dst)
}
