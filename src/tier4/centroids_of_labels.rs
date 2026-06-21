use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier3;
use crate::types::{DType, MType};

/// Determines the centroids of all labels in a label image or image stack.
///
/// Writes the resulting coordinates into a point list image of dimensions
/// `n x 3 x 1`, where `n` is the number of labels and the rows are x, y, and z.
pub fn centroids_of_labels(
    device: &DeviceArc,
    label_image: &ArrayPtr,
    centroids_coordinates: Option<ArrayPtr>,
    include_background: bool,
) -> Result<ArrayPtr> {
    let props = if include_background {
        tier3::statistics_of_background_and_labelled_pixels(
            device,
            Some(label_image),
            Some(label_image),
        )?
    } else {
        tier3::statistics_of_labelled_pixels(device, Some(label_image), Some(label_image))?
    };

    let centroid_x = props
        .get("centroid_x")
        .ok_or_else(|| CleError::Other("centroids_of_labels: Missing centroid_x.".to_string()))?;
    let centroid_y = props
        .get("centroid_y")
        .ok_or_else(|| CleError::Other("centroids_of_labels: Missing centroid_y.".to_string()))?;
    let centroid_z = props
        .get("centroid_z")
        .ok_or_else(|| CleError::Other("centroids_of_labels: Missing centroid_z.".to_string()))?;
    let nb_labels = centroid_x.len();

    let centroids_coordinates = match centroids_coordinates {
        Some(dst) => dst,
        None => Array::create(nb_labels, 3, 1, 1, DType::Float, MType::Buffer, device)?,
    };

    {
        let dst = centroids_coordinates.lock().unwrap();
        if dst.width() != nb_labels || dst.height() != 3 {
            return Err(CleError::Other(format!(
                "centroids_of_labels: Provided output array has wrong dimensions.{}x{}x1 instead of {}x3x1",
                dst.width(),
                dst.height(),
                nb_labels
            )));
        }
        if dst.dtype() != DType::Float {
            return Err(CleError::Other(
                "centroids_of_labels: Provided output array has wrong data type. Expected dtype==FLOAT."
                    .to_string(),
            ));
        }
    }

    let mut coordinates = Vec::with_capacity(nb_labels * 3);
    coordinates.extend_from_slice(centroid_x);
    coordinates.extend_from_slice(centroid_y);
    coordinates.extend_from_slice(centroid_z);
    centroids_coordinates
        .lock()
        .unwrap()
        .write_from_typed(&coordinates)?;

    Ok(centroids_coordinates)
}
