use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;
use crate::tier3;
use crate::types::{MType, LABEL};

/// Remove labels whose associated map values are outside `[min_value, max_value]`.
pub fn remove_labels_with_map_values_out_of_range(
    device: &DeviceArc,
    src: &ArrayPtr,
    values: &ArrayPtr,
    dst: Option<ArrayPtr>,
    min_value: f32,
    max_value: f32,
) -> Result<ArrayPtr> {
    let above = tier1::greater_constant(device, values, None, max_value)?;
    let below = tier1::smaller_constant(device, values, None, min_value)?;
    let flaglist = Array::create(
        values.lock().unwrap().size(),
        1,
        1,
        1,
        LABEL,
        MType::Buffer,
        device,
    )?;
    tier1::binary_or(device, &below, &above, Some(flaglist.clone()))?;
    tier3::remove_labels(device, src, &flaglist, dst)
}

/// Remove labels whose associated map values are inside `[min_value, max_value]`.
pub fn remove_labels_with_map_values_within_range(
    device: &DeviceArc,
    src: &ArrayPtr,
    values: &ArrayPtr,
    dst: Option<ArrayPtr>,
    min_value: f32,
    max_value: f32,
) -> Result<ArrayPtr> {
    let above = tier1::greater_or_equal_constant(device, values, None, min_value)?;
    let below = tier1::smaller_or_equal_constant(device, values, None, max_value)?;
    let flaglist = Array::create(
        values.lock().unwrap().size(),
        1,
        1,
        1,
        LABEL,
        MType::Buffer,
        device,
    )?;
    tier1::binary_and(device, &below, &above, Some(flaglist.clone()))?;
    tier3::remove_labels(device, src, &flaglist, dst)
}

/// Deprecated CLIc alias for [`remove_labels_with_map_values_out_of_range`].
pub fn exclude_labels_with_map_values_out_of_range(
    device: &DeviceArc,
    values_map: &ArrayPtr,
    label_map_input: &ArrayPtr,
    dst: Option<ArrayPtr>,
    minimum_value_range: f32,
    maximum_value_range: f32,
) -> Result<ArrayPtr> {
    remove_labels_with_map_values_out_of_range(
        device,
        label_map_input,
        values_map,
        dst,
        minimum_value_range,
        maximum_value_range,
    )
}

/// Deprecated CLIc alias for [`remove_labels_with_map_values_within_range`].
pub fn exclude_labels_with_map_values_within_range(
    device: &DeviceArc,
    values_map: &ArrayPtr,
    label_map_input: &ArrayPtr,
    dst: Option<ArrayPtr>,
    minimum_value_range: f32,
    maximum_value_range: f32,
) -> Result<ArrayPtr> {
    remove_labels_with_map_values_within_range(
        device,
        label_map_input,
        values_map,
        dst,
        minimum_value_range,
        maximum_value_range,
    )
}
