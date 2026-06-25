use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::types::DType;

pub fn generate_proximal_neighbors_matrix(
    device: &DeviceArc,
    src_matrix: &ArrayPtr,
    dst_matrix: Option<ArrayPtr>,
    mut min_distance: f32,
    mut max_distance: f32,
) -> Result<ArrayPtr> {
    let dst_matrix = tier0::create_like(src_matrix, dst_matrix, DType::Uint8, device)?;
    min_distance = min_distance.max(0.0);
    max_distance = if max_distance < 0.0 {
        f32::MAX
    } else {
        max_distance
    };

    let above_min_distance =
        tier1::greater_or_equal_constant(device, src_matrix, None, min_distance)?;
    let below_max_distance =
        tier1::smaller_or_equal_constant(device, src_matrix, None, max_distance)?;

    tier1::binary_and(
        device,
        &above_min_distance,
        &below_max_distance,
        Some(dst_matrix.clone()),
    )?;
    tier1::set_where_x_equals_y(device, &dst_matrix, 0.0)?;
    tier1::set_row(device, &dst_matrix, 0, 0.0)?;
    tier1::set_column(device, &dst_matrix, 0, 0.0)?;
    Ok(dst_matrix)
}
