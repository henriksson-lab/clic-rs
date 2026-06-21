use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier5;

/// Remove labels smaller than `minimum_size`.
///
/// Mirrors CLIc's `remove_small_labels_func`.
pub fn remove_small_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    minimum_size: f32,
) -> Result<ArrayPtr> {
    tier5::filter_label_by_size(device, src, dst, minimum_size, f32::MAX)
}

/// Remove labels larger than `maximum_size`.
///
/// Mirrors CLIc's `remove_large_labels_func`.
pub fn remove_large_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    maximum_size: f32,
) -> Result<ArrayPtr> {
    tier5::filter_label_by_size(device, src, dst, 0.0, maximum_size)
}

/// Deprecated CLIc alias for [`remove_small_labels`].
pub fn exclude_small_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    maximum_size: f32,
) -> Result<ArrayPtr> {
    remove_small_labels(device, src, dst, maximum_size)
}

/// Deprecated CLIc alias for [`remove_large_labels`].
pub fn exclude_large_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    minimum_size: f32,
) -> Result<ArrayPtr> {
    remove_large_labels(device, src, dst, minimum_size)
}
