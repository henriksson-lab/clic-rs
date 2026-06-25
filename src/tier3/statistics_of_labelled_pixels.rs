use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier1;
use crate::types::{DType, LABEL};

pub type StatisticsMap = crate::statistics::StatisticsMap;

pub fn statistics_of_labelled_pixels(
    device: &DeviceArc,
    intensity: Option<&ArrayPtr>,
    label: Option<&ArrayPtr>,
) -> Result<StatisticsMap> {
    let mut intensity = intensity.cloned();
    let mut label = label.cloned();

    if intensity.is_none() && label.is_none() {
        return Err(CleError::Other(
            "Error: no intensity nor label was provided to the 'statistics_of_labelled_pixels' function.".to_string(),
        ));
    }
    if label.is_none() {
        let new_label =
            crate::tier0::create_like(intensity.as_ref().unwrap(), None, LABEL, device)?;
        new_label.lock().unwrap().fill(1.0)?;
        label = Some(new_label);
    }
    if intensity.is_none() {
        let new_intensity =
            crate::tier0::create_like(label.as_ref().unwrap(), None, DType::Float, device)?;
        tier1::copy(device, label.as_ref().unwrap(), Some(new_intensity.clone()))?;
        intensity = Some(new_intensity);
    }

    crate::statistics::compute_statistics_per_labels(
        device,
        label.as_ref().unwrap(),
        intensity.as_ref().unwrap(),
    )
}

pub fn statistics_of_background_and_labelled_pixels(
    device: &DeviceArc,
    intensity: Option<&ArrayPtr>,
    label: Option<&ArrayPtr>,
) -> Result<StatisticsMap> {
    let mut intensity = intensity.cloned();
    let mut label = label.cloned();

    if intensity.is_none() && label.is_none() {
        return Err(CleError::Other(
            "Error: no intensity nor label was provided to the 'statistics_of_labelled_pixels' function.".to_string(),
        ));
    }
    if label.is_none() {
        let new_label =
            crate::tier0::create_like(intensity.as_ref().unwrap(), None, LABEL, device)?;
        new_label.lock().unwrap().fill(1.0)?;
        label = Some(new_label);
    }
    if intensity.is_none() {
        let new_intensity =
            crate::tier0::create_like(label.as_ref().unwrap(), None, DType::Float, device)?;
        tier1::copy(device, label.as_ref().unwrap(), Some(new_intensity.clone()))?;
        intensity = Some(new_intensity);
    }
    let temp = tier1::add_image_and_scalar(device, label.as_ref().unwrap(), None, 1.0)?;
    let mut props =
        statistics_of_labelled_pixels(device, Some(intensity.as_ref().unwrap()), Some(&temp))?;
    let mut labels = props["label"].clone();
    for i in 0..labels.len() {
        labels[i] -= 1.0;
    }
    props.insert("label".to_string(), labels);
    Ok(props)
}
