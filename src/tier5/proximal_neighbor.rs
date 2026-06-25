use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::types::DType;
use crate::{tier0, tier1, tier2, tier4};

/// Count labels whose centroids are within a distance range of each other.
///
/// Returns a vector containing one count per label. Negative `max_distance`
/// means no upper bound, and negative `min_distance` is clamped to zero.
/// Mirrors CLIc's `proximal_neighbor_count_func`.
pub fn proximal_neighbor_count(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    mut min_distance: f32,
    mut max_distance: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Uint32, device)?;

    min_distance = min_distance.max(0.0);
    max_distance = if max_distance < 0.0 {
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

    tier2::count_touching_neighbors(device, &touch_matrix, Some(dst.clone()), false)?;
    tier1::set_column(device, &dst, 0, 0.0)?;

    Ok(dst)
}
