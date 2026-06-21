use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier0;
use crate::tier3;
use crate::tier4;
use crate::types::{MType, LABEL};

/// Remove labels outside the inclusive size range.
///
/// Mirrors CLIc's `filter_label_by_size_func`.
pub fn filter_label_by_size(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    minimum_size: f32,
    maximum_size: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    let stats = tier3::statistics_of_background_and_labelled_pixels(device, None, Some(src))?;
    let areas = stats.get("area").ok_or_else(|| {
        CleError::Other("filter_label_by_size: Missing area statistics.".to_string())
    })?;
    let list_of_area = Array::create_with_data(
        areas.len(),
        1,
        1,
        1,
        MType::Buffer,
        areas.as_slice(),
        device,
    )?;
    tier4::remove_labels_with_map_values_out_of_range(
        device,
        src,
        &list_of_area,
        Some(dst),
        minimum_size,
        maximum_size,
    )
}

/// CLIc alias for [`filter_label_by_size`].
pub fn exclude_labels_outside_size_range(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    minimum_size: f32,
    maximum_size: f32,
) -> Result<ArrayPtr> {
    filter_label_by_size(device, src, dst, minimum_size, maximum_size)
}
