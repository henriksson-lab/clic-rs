use crate::array::{pull, Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::types::{DType, LABEL};
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

    let centroids = tier4::centroids_of_labels(device, src, None, true)?;
    let (width, mtype) = {
        let centroids = centroids.lock().unwrap();
        (centroids.width(), centroids.mtype())
    };
    let centroid_values = pull::<f32>(&centroids)?;

    let mut labelled_pointlist = vec![0.0f32; width * 4];
    for label_id in 0..width {
        for coordinate in 0..3 {
            let value = centroid_values[label_id + width * coordinate];
            labelled_pointlist[label_id + width * coordinate] =
                if value.is_finite() { value } else { -1.0 };
        }
        labelled_pointlist[label_id + width * 3] = label_id as f32;
    }
    if width > 0 {
        for row in 0..4 {
            labelled_pointlist[width * row] = -1.0;
        }
    }

    let label_pos = Array::create(width, 4, 1, 2, DType::Float, mtype, device)?;
    label_pos
        .lock()
        .unwrap()
        .write_from_typed(&labelled_pointlist)?;

    tier1::write_values_to_positions(device, &label_pos, Some(dst))
}
