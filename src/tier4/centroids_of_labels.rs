use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier3;
use crate::types::{DType, MType};

pub fn centroids_of_labels(
    device: &DeviceArc,
    label_image: &ArrayPtr,
    centroids_coordinates: Option<ArrayPtr>,
    include_background: bool,
) -> Result<ArrayPtr> {
    let props;
    if include_background {
        props = tier3::statistics_of_background_and_labelled_pixels(
            device,
            Some(label_image),
            Some(label_image),
        )?;
    } else {
        props = tier3::statistics_of_labelled_pixels(device, Some(label_image), Some(label_image))?;
    }

    let centroid_x = props["centroid_x"].clone();
    let centroid_y = props["centroid_y"].clone();
    let centroid_z = props["centroid_z"].clone();
    let nb_labels = centroid_x.len();

    let mut centroids_coordinates = centroids_coordinates;
    if centroids_coordinates.is_none() {
        centroids_coordinates = Some(Array::create(
            nb_labels,
            3,
            1,
            1,
            DType::Float,
            MType::Buffer,
            device,
        )?);
    }
    let centroids_coordinates = centroids_coordinates.unwrap();

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

    {
        let centroids_coordinates = centroids_coordinates.lock().unwrap();
        centroids_coordinates.write_from_region(&centroid_x, [nb_labels, 1, 1], [0, 0, 0])?;
        centroids_coordinates.write_from_region(&centroid_y, [nb_labels, 1, 1], [0, 1, 0])?;
        centroids_coordinates.write_from_region(&centroid_z, [nb_labels, 1, 1], [0, 2, 0])?;
    }

    Ok(centroids_coordinates)
}
