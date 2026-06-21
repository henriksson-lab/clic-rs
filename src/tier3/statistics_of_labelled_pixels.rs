use std::collections::HashMap;

use crate::array::{pull, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier1;
use crate::types::{DType, LABEL};

pub type StatisticsMap = HashMap<String, Vec<f32>>;

#[derive(Clone)]
struct Accumulator {
    area: f32,
    sum_x: f32,
    sum_y: f32,
    sum_z: f32,
    sum_intensity: f32,
    sum_intensity_x: f32,
    sum_intensity_y: f32,
    sum_intensity_z: f32,
    min_intensity: f32,
    max_intensity: f32,
    bbox_min_x: f32,
    bbox_min_y: f32,
    bbox_min_z: f32,
    bbox_max_x: f32,
    bbox_max_y: f32,
    bbox_max_z: f32,
    sum_distance_to_centroid: f32,
    sum_distance_to_mass_center: f32,
    sum_squared_difference: f32,
    max_distance_to_centroid: f32,
    max_distance_to_mass_center: f32,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            area: 0.0,
            sum_x: 0.0,
            sum_y: 0.0,
            sum_z: 0.0,
            sum_intensity: 0.0,
            sum_intensity_x: 0.0,
            sum_intensity_y: 0.0,
            sum_intensity_z: 0.0,
            min_intensity: 0.0,
            max_intensity: 0.0,
            bbox_min_x: 0.0,
            bbox_min_y: 0.0,
            bbox_min_z: 0.0,
            bbox_max_x: 0.0,
            bbox_max_y: 0.0,
            bbox_max_z: 0.0,
            sum_distance_to_centroid: 0.0,
            sum_distance_to_mass_center: 0.0,
            sum_squared_difference: 0.0,
            max_distance_to_centroid: 0.0,
            max_distance_to_mass_center: 0.0,
        }
    }
}

fn pull_as_f32(arr: &ArrayPtr) -> Result<Vec<f32>> {
    match arr.lock().unwrap().dtype() {
        DType::Float => pull::<f32>(arr),
        DType::Uint32 => Ok(pull::<u32>(arr)?.into_iter().map(|v| v as f32).collect()),
        DType::Int32 => Ok(pull::<i32>(arr)?.into_iter().map(|v| v as f32).collect()),
        DType::Uint8 => Ok(pull::<u8>(arr)?.into_iter().map(|v| v as f32).collect()),
        DType::Int8 => Ok(pull::<i8>(arr)?.into_iter().map(|v| v as f32).collect()),
        DType::Uint16 => Ok(pull::<u16>(arr)?.into_iter().map(|v| v as f32).collect()),
        DType::Int16 => Ok(pull::<i16>(arr)?.into_iter().map(|v| v as f32).collect()),
        dtype => Err(CleError::Other(format!(
            "statistics: unsupported intensity dtype {:?}",
            dtype
        ))),
    }
}

fn pull_labels(arr: &ArrayPtr) -> Result<Vec<u32>> {
    match arr.lock().unwrap().dtype() {
        DType::Uint32 => pull::<u32>(arr),
        DType::Int32 => Ok(pull::<i32>(arr)?
            .into_iter()
            .map(|v| v.max(0) as u32)
            .collect()),
        DType::Uint16 => Ok(pull::<u16>(arr)?.into_iter().map(|v| v as u32).collect()),
        DType::Int16 => Ok(pull::<i16>(arr)?
            .into_iter()
            .map(|v| v.max(0) as u32)
            .collect()),
        DType::Uint8 => Ok(pull::<u8>(arr)?.into_iter().map(|v| v as u32).collect()),
        dtype => Err(CleError::Other(format!(
            "statistics: unsupported label dtype {:?}",
            dtype
        ))),
    }
}

fn compute_statistics(
    label: &ArrayPtr,
    intensity: &ArrayPtr,
    include_background: bool,
) -> Result<StatisticsMap> {
    let labels = pull_labels(label)?;
    let intensities = pull_as_f32(intensity)?;
    let (width, height, depth) = {
        let label = label.lock().unwrap();
        (label.width(), label.height(), label.depth())
    };
    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let start_label = if include_background { 0 } else { 1 };
    let mut acc = vec![Accumulator::default(); max_label + 1];

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let index = z * width * height + y * width + x;
                let label_id = labels[index] as usize;
                if label_id == 0 && !include_background {
                    continue;
                }
                let value = intensities[index];
                let a = &mut acc[label_id];
                if a.area == 0.0 {
                    a.min_intensity = value;
                    a.max_intensity = value;
                    a.bbox_min_x = x as f32;
                    a.bbox_max_x = x as f32;
                    a.bbox_min_y = y as f32;
                    a.bbox_max_y = y as f32;
                    a.bbox_min_z = z as f32;
                    a.bbox_max_z = z as f32;
                } else {
                    a.min_intensity = a.min_intensity.min(value);
                    a.max_intensity = a.max_intensity.max(value);
                    a.bbox_min_x = a.bbox_min_x.min(x as f32);
                    a.bbox_max_x = a.bbox_max_x.max(x as f32);
                    a.bbox_min_y = a.bbox_min_y.min(y as f32);
                    a.bbox_max_y = a.bbox_max_y.max(y as f32);
                    a.bbox_min_z = a.bbox_min_z.min(z as f32);
                    a.bbox_max_z = a.bbox_max_z.max(z as f32);
                }
                a.area += 1.0;
                a.sum_x += x as f32;
                a.sum_y += y as f32;
                a.sum_z += z as f32;
                a.sum_intensity += value;
                a.sum_intensity_x += x as f32 * value;
                a.sum_intensity_y += y as f32 * value;
                a.sum_intensity_z += z as f32 * value;
            }
        }
    }

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let index = z * width * height + y * width + x;
                let label_id = labels[index] as usize;
                if label_id == 0 && !include_background {
                    continue;
                }
                let a = &mut acc[label_id];
                if a.area == 0.0 {
                    continue;
                }
                let centroid_x = a.sum_x / a.area;
                let centroid_y = a.sum_y / a.area;
                let centroid_z = a.sum_z / a.area;
                let mass_center_x = a.sum_intensity_x / a.sum_intensity;
                let mass_center_y = a.sum_intensity_y / a.sum_intensity;
                let mass_center_z = a.sum_intensity_z / a.sum_intensity;
                let mean_intensity = a.sum_intensity / a.area;
                let value = intensities[index];
                let centroid_distance = ((x as f32 - centroid_x).powi(2)
                    + (y as f32 - centroid_y).powi(2)
                    + (z as f32 - centroid_z).powi(2))
                .sqrt();
                let mass_center_distance = ((x as f32 - mass_center_x).powi(2)
                    + (y as f32 - mass_center_y).powi(2)
                    + (z as f32 - mass_center_z).powi(2))
                .sqrt();
                a.sum_distance_to_centroid += centroid_distance;
                a.sum_distance_to_mass_center += mass_center_distance;
                a.sum_squared_difference += (value - mean_intensity).powi(2) / a.area;
                a.max_distance_to_centroid = a.max_distance_to_centroid.max(centroid_distance);
                a.max_distance_to_mass_center =
                    a.max_distance_to_mass_center.max(mass_center_distance);
            }
        }
    }

    let labels_out = (start_label..=max_label)
        .map(|v| v as f32)
        .collect::<Vec<_>>();
    let mut props = StatisticsMap::new();
    props.insert("label".to_string(), labels_out);

    macro_rules! collect_prop {
        ($name:expr, $expr:expr) => {{
            let mut values = Vec::new();
            for item in acc.iter().take(max_label + 1).skip(start_label) {
                values.push($expr(item));
            }
            props.insert($name.to_string(), values);
        }};
    }

    collect_prop!("area", |a: &Accumulator| a.area);
    collect_prop!("min_intensity", |a: &Accumulator| a.min_intensity);
    collect_prop!("max_intensity", |a: &Accumulator| a.max_intensity);
    collect_prop!("sum_intensity", |a: &Accumulator| a.sum_intensity);
    collect_prop!("mean_intensity", |a: &Accumulator| a.sum_intensity / a.area);
    collect_prop!("sum_x", |a: &Accumulator| a.sum_x);
    collect_prop!("sum_y", |a: &Accumulator| a.sum_y);
    collect_prop!("sum_z", |a: &Accumulator| a.sum_z);
    collect_prop!("centroid_x", |a: &Accumulator| a.sum_x / a.area);
    collect_prop!("centroid_y", |a: &Accumulator| a.sum_y / a.area);
    collect_prop!("centroid_z", |a: &Accumulator| a.sum_z / a.area);
    collect_prop!("sum_intensity_times_x", |a: &Accumulator| a.sum_intensity_x);
    collect_prop!("sum_intensity_times_y", |a: &Accumulator| a.sum_intensity_y);
    collect_prop!("sum_intensity_times_z", |a: &Accumulator| a.sum_intensity_z);
    collect_prop!("mass_center_x", |a: &Accumulator| a.sum_intensity_x
        / a.sum_intensity);
    collect_prop!("mass_center_y", |a: &Accumulator| a.sum_intensity_y
        / a.sum_intensity);
    collect_prop!("mass_center_z", |a: &Accumulator| a.sum_intensity_z
        / a.sum_intensity);
    collect_prop!("bbox_min_x", |a: &Accumulator| a.bbox_min_x);
    collect_prop!("bbox_min_y", |a: &Accumulator| a.bbox_min_y);
    collect_prop!("bbox_min_z", |a: &Accumulator| a.bbox_min_z);
    collect_prop!("bbox_max_x", |a: &Accumulator| a.bbox_max_x);
    collect_prop!("bbox_max_y", |a: &Accumulator| a.bbox_max_y);
    collect_prop!("bbox_max_z", |a: &Accumulator| a.bbox_max_z);
    collect_prop!("bbox_width", |a: &Accumulator| a.bbox_max_x - a.bbox_min_x
        + 1.0);
    collect_prop!("bbox_height", |a: &Accumulator| a.bbox_max_y - a.bbox_min_y
        + 1.0);
    collect_prop!("bbox_depth", |a: &Accumulator| a.bbox_max_z - a.bbox_min_z
        + 1.0);
    collect_prop!("sum_distance_to_centroid", |a: &Accumulator| a
        .sum_distance_to_centroid);
    collect_prop!("mean_distance_to_centroid", |a: &Accumulator| a
        .sum_distance_to_centroid
        / a.area);
    collect_prop!("sum_distance_to_mass_center", |a: &Accumulator| a
        .sum_distance_to_mass_center);
    collect_prop!("mean_distance_to_mass_center", |a: &Accumulator| a
        .sum_distance_to_mass_center
        / a.area);
    collect_prop!("standard_deviation_intensity", |a: &Accumulator| a
        .sum_squared_difference
        .sqrt());
    collect_prop!("max_distance_to_centroid", |a: &Accumulator| a
        .max_distance_to_centroid);
    collect_prop!("max_distance_to_mass_center", |a: &Accumulator| a
        .max_distance_to_mass_center);
    collect_prop!("mean_max_distance_to_centroid_ratio", |a: &Accumulator| {
        a.max_distance_to_centroid / (a.sum_distance_to_centroid / a.area)
    });
    collect_prop!(
        "mean_max_distance_to_mass_center_ratio",
        |a: &Accumulator| {
            a.max_distance_to_mass_center / (a.sum_distance_to_mass_center / a.area)
        }
    );

    Ok(props)
}

pub fn statistics_of_labelled_pixels(
    device: &DeviceArc,
    intensity: Option<&ArrayPtr>,
    label: Option<&ArrayPtr>,
) -> Result<StatisticsMap> {
    match (intensity, label) {
        (None, None) => Err(CleError::Other(
            "Error: no intensity nor label was provided to the 'statistics_of_labelled_pixels' function.".to_string(),
        )),
        (Some(intensity), None) => {
            let label = crate::tier0::create_like(intensity, None, LABEL, device)?;
            label.lock().unwrap().fill(1.0)?;
            compute_statistics(&label, intensity, false)
        }
        (None, Some(label)) => {
            let intensity = crate::tier0::create_like(label, None, DType::Float, device)?;
            tier1::copy(device, label, Some(intensity.clone()))?;
            compute_statistics(label, &intensity, false)
        }
        (Some(intensity), Some(label)) => compute_statistics(label, intensity, false),
    }
}

pub fn statistics_of_background_and_labelled_pixels(
    device: &DeviceArc,
    intensity: Option<&ArrayPtr>,
    label: Option<&ArrayPtr>,
) -> Result<StatisticsMap> {
    match (intensity, label) {
        (None, None) => Err(CleError::Other(
            "Error: no intensity nor label was provided to the 'statistics_of_labelled_pixels' function.".to_string(),
        )),
        (Some(intensity), None) => {
            let label = crate::tier0::create_like(intensity, None, LABEL, device)?;
            label.lock().unwrap().fill(1.0)?;
            compute_statistics(&label, intensity, true)
        }
        (None, Some(label)) => {
            let intensity = crate::tier0::create_like(label, None, DType::Float, device)?;
            tier1::copy(device, label, Some(intensity.clone()))?;
            compute_statistics(label, &intensity, true)
        }
        (Some(intensity), Some(label)) => compute_statistics(label, intensity, true),
    }
}
