use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::types::{MType, LABEL};
use crate::{tier1, tier2, tier3, tier4};

/// Merge touching labels of a label image and relabel the result sequentially.
///
/// Mirrors CLIc's `merge_touching_labels_func`.
pub fn merge_touching_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    // generate touching matrix
    let touching_matrix = tier3::generate_touch_matrix(device, src, None)?;
    tier1::set_column(device, &touching_matrix, 0, 0.0)?;
    tier1::set_row(device, &touching_matrix, 0, 0.0)?;

    // check if there are touching labels
    if tier2::maximum_of_all_pixels(device, &touching_matrix)? == 0.0 {
        return Ok(src.clone());
    }

    let dst = tier0::create_like(src, dst, LABEL, device)?;

    // prepare the touching matrix for processing
    tier1::set_where_x_equals_y(device, &touching_matrix, 1.0)?;
    tier1::set_column(device, &touching_matrix, 0, 0.0)?;
    tier1::set_row(device, &touching_matrix, 0, 0.0)?;

    // make a touch-matrix where intensities correspond to the label-ID
    let (width, height) = {
        let matrix = touching_matrix.lock().unwrap();
        (matrix.width(), matrix.height())
    };
    let mut label_id_vector = Array::create(1, height, 1, 1, LABEL, MType::Buffer, device)?;
    tier1::set_ramp_y(device, &label_id_vector)?;

    let touching_matrix_id = Array::create(width, height, 1, 2, LABEL, MType::Buffer, device)?;
    tier1::multiply_images(
        device,
        &touching_matrix,
        &label_id_vector,
        Some(touching_matrix_id.clone()),
    )?;

    // new list of labels corresponding to maximum x projection of touching matrix
    // e.g. if label 2 and 3 are touching, both will have value to 3 in the list
    label_id_vector = tier1::maximum_y_projection(device, &touching_matrix_id, None)?;

    // renumber labels sequentially and replace intensities in the label image with the new labels
    let new_labels = tier4::relabel_sequential(device, &label_id_vector, None, 4096)?;
    tier1::replace_intensities(device, src, &new_labels, Some(dst.clone()))?;

    // Recall again to make sure all are merged (do we want to do this?)
    merge_touching_labels(device, &dst, None)
}
