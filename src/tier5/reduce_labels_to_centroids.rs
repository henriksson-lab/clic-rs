use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::types::{DType, MType, LABEL};
use crate::{tier0, tier1, tier4};

/// Takes a label map and reduces each label to its centroid.
///
/// Mirrors CLIc's `reduce_labels_to_centroids_func`.
pub fn reduce_labels_to_centroids(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    dst.lock().unwrap().fill(0.0)?;

    let pos = tier4::centroids_of_labels(device, src, None, true)?;

    let label_pos = Array::create(
        pos.lock().unwrap().width(),
        4,
        1,
        2,
        DType::Float,
        MType::Buffer,
        device,
    )?;
    tier1::set_ramp_x(device, &label_pos)?;
    let width = pos.lock().unwrap().width();
    pos.lock()
        .unwrap()
        .copy_to_region(&label_pos, [width, 3, 1], [0, 0, 0], [0, 0, 0])?;

    tier1::set_column(device, &label_pos, 0, -1.0)?;
    tier1::nan_to_num(
        device,
        &label_pos,
        Some(label_pos.clone()),
        -1.0,
        -1.0,
        -1.0,
    )?;

    tier1::write_values_to_positions(device, &label_pos, Some(dst))
}
