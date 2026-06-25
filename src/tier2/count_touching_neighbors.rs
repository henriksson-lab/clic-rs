use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::types::DType;

/// Count touching neighbors for each label in a touch matrix.
///
/// Mirrors CLIc's `count_touching_neighbors_func`.
pub fn count_touching_neighbors(
    device: &DeviceArc,
    touch_matrix: &ArrayPtr,
    touching_neighbors_count_destination: Option<ArrayPtr>,
    ignore_background: bool,
) -> Result<ArrayPtr> {
    let touching_neighbors_count_destination = tier0::create_vector(
        touch_matrix,
        touching_neighbors_count_destination,
        touch_matrix.lock().unwrap().width(),
        DType::Uint32,
        device,
    )?;
    let bin_matrix = tier1::greater_constant(device, touch_matrix, None, 0.0)?;

    if ignore_background {
        tier1::set_row(device, &bin_matrix, 0, 0.0)?;
        tier1::set_column(device, &bin_matrix, 0, 0.0)?;
        tier1::set_where_x_equals_y(device, &bin_matrix, 0.0)?;
    }

    tier1::sum_y_projection(
        device,
        &bin_matrix,
        Some(touching_neighbors_count_destination),
    )
}
