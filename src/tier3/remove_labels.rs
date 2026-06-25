use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier0;
use crate::tier1;
use crate::types::{DType, MType, LABEL};

/// Remove labels from a label map and renumber the remaining labels.
///
/// `list` is a uint32 flag vector beginning with the background flag, followed
/// by flags for labels 1, 2, and so on. Non-zero flags are removed; zero flags
/// are kept and renumbered sequentially.
pub fn remove_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    list: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let list_size = {
        let list = list.lock().unwrap();
        if list.dtype() != LABEL {
            return Err(CleError::Other(
                "remove_labels: label list must be of type uint32".to_string(),
            ));
        }
        list.size()
    };

    let mut labels_list = vec![0_u32; list_size];
    list.lock().unwrap().read_to(&mut labels_list)?;
    labels_list[0] = 0;
    let mut count = 1;
    for label in labels_list.iter_mut().take(list_size).skip(1) {
        if *label == 0 {
            *label = count;
            count += 1;
        } else {
            *label = 0;
        }
    }

    let index_list =
        Array::create_with_data(list_size, 1, 1, 1, MType::Buffer, &labels_list, device)?;
    tier1::replace_values(device, src, &index_list, Some(dst.clone()))?;
    Ok(dst)
}

/// Alias for [`remove_labels`].
pub fn exclude_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    list: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    remove_labels(device, src, list, dst)
}
