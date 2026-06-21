use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier3;

pub type StatisticsMap = tier3::StatisticsMap;

pub fn compute_statistics_per_labels(
    device: &DeviceArc,
    label: &ArrayPtr,
    intensity: &ArrayPtr,
) -> Result<StatisticsMap> {
    tier3::statistics_of_labelled_pixels(device, Some(intensity), Some(label))
}

pub fn _statistics_per_label(
    _device: &DeviceArc,
    _label: &ArrayPtr,
    _intensity: &ArrayPtr,
    _nb_labels: i32,
) -> Result<ArrayPtr> {
    Err(CleError::Other(
        "_statistics_per_label: low-level GPU statistics stack helper is not implemented; use compute_statistics_per_labels instead.".to_string(),
    ))
}

pub fn _std_per_label(
    _device: &DeviceArc,
    _statistics: &ArrayPtr,
    _label: &ArrayPtr,
    _intensity: &ArrayPtr,
    _nb_labels: i32,
) -> Result<ArrayPtr> {
    Err(CleError::Other(
        "_std_per_label: low-level GPU standard-deviation stack helper is not implemented; use compute_statistics_per_labels instead.".to_string(),
    ))
}
