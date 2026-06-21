use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::tier3;
use crate::types::{DType, MType};

type TouchingNeighborOp =
    fn(&DeviceArc, &ArrayPtr, &ArrayPtr, Option<ArrayPtr>) -> Result<ArrayPtr>;

/// Map a per-label statistic back onto a label image.
///
/// Mirrors CLIc's `parametric_map_func`.
pub fn parametric_map(
    device: &DeviceArc,
    labels: &ArrayPtr,
    intensity: Option<&ArrayPtr>,
    property: &str,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let intensity = intensity.unwrap_or(labels);
    let dst = tier0::create_like(labels, dst, DType::Float, device)?;
    let props =
        tier3::statistics_of_background_and_labelled_pixels(device, Some(intensity), Some(labels))?;
    let property = property.to_lowercase();
    let vector = props.get(&property).ok_or_else(|| {
        CleError::Other(format!("Property '{}' not found in statistics", property))
    })?;
    let values = Array::create_with_data(
        vector.len(),
        1,
        1,
        1,
        MType::Buffer,
        vector.as_slice(),
        device,
    )?;
    tier1::set_column(device, &values, 0, 0.0)?;
    tier1::replace_values(device, labels, &values, Some(dst))
}

/// Map label area/volume back onto a label image.
///
/// Mirrors CLIc's `pixel_count_map_func`.
pub fn pixel_count_map(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    parametric_map(device, src, None, "area", dst)
}

/// Deprecated CLIc alias for [`pixel_count_map`].
pub fn label_pixel_count_map(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    pixel_count_map(device, src, dst)
}

/// Mirrors CLIc's `extension_ratio_map_func`.
pub fn extension_ratio_map(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    parametric_map(
        device,
        src,
        None,
        "mean_max_distance_to_centroid_ratio",
        dst,
    )
}

/// Mirrors CLIc's `mean_extension_map_func`.
pub fn mean_extension_map(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    parametric_map(device, src, None, "mean_distance_to_centroid", dst)
}

/// Mirrors CLIc's `maximum_extension_map_func`.
pub fn maximum_extension_map(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    parametric_map(device, src, None, "max_distance_to_centroid", dst)
}

/// Mirrors CLIc's `mean_intensity_map_func`.
pub fn mean_intensity_map(
    device: &DeviceArc,
    src: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    parametric_map(device, labels, Some(src), "mean_intensity", dst)
}

/// Deprecated CLIc alias for [`mean_intensity_map`].
pub fn label_mean_intensity_map(
    device: &DeviceArc,
    src: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    mean_intensity_map(device, src, labels, dst)
}

/// Mirrors CLIc's `minimum_intensity_map_func`.
pub fn minimum_intensity_map(
    device: &DeviceArc,
    src: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    parametric_map(device, labels, Some(src), "min_intensity", dst)
}

/// Mirrors CLIc's `maximum_intensity_map_func`.
pub fn maximum_intensity_map(
    device: &DeviceArc,
    src: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    parametric_map(device, labels, Some(src), "max_intensity", dst)
}

/// Mirrors CLIc's `standard_deviation_intensity_map_func`.
pub fn standard_deviation_intensity_map(
    device: &DeviceArc,
    src: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    parametric_map(
        device,
        labels,
        Some(src),
        "standard_deviation_intensity",
        dst,
    )
}

/// Mirrors CLIc's `touching_neighbor_count_map_func`.
pub fn touching_neighbor_count_map(
    device: &DeviceArc,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(labels, dst, DType::Float, device)?;
    let touch_matrix = tier3::generate_touch_matrix(device, labels, None)?;
    tier1::set_column(device, &touch_matrix, 0, 0.0)?;
    let nb_touching_neighbors = tier2::count_touching_neighbors(device, &touch_matrix, None, true)?;
    tier1::replace_values(device, labels, &nb_touching_neighbors, Some(dst))
}

fn apply_touching_neighbors_operation(
    device: &DeviceArc,
    map: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
    ignore_background: bool,
    operation: TouchingNeighborOp,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(map, dst, DType::Float, device)?;
    if radius < 1 {
        tier1::copy(device, map, Some(dst.clone()))?;
        return Ok(dst);
    }

    let mut touch_matrix = tier3::generate_touch_matrix(device, labels, None)?;
    if ignore_background {
        tier1::set_column(device, &touch_matrix, 0, 0.0)?;
    }
    for _ in 1..radius {
        touch_matrix = tier3::generate_touch_matrix(device, &touch_matrix, None)?;
        if ignore_background {
            tier1::set_column(device, &touch_matrix, 0, 0.0)?;
        }
    }

    let values = tier3::read_map_values(device, map, labels, None)?;
    let new_values = operation(device, &values, &touch_matrix, None)?;
    tier1::set_column(device, &new_values, 0, 0.0)?;
    tier1::replace_intensities(device, labels, &new_values, Some(dst))
}

/// Mirrors CLIc's `mean_of_touching_neighbors_map_func`.
pub fn mean_of_touching_neighbors_map(
    device: &DeviceArc,
    map: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
    ignore_background: bool,
) -> Result<ArrayPtr> {
    apply_touching_neighbors_operation(
        device,
        map,
        labels,
        dst,
        radius,
        ignore_background,
        tier1::mean_of_touching_neighbors,
    )
}

/// Mirrors CLIc's `minimum_of_touching_neighbors_map_func`.
pub fn minimum_of_touching_neighbors_map(
    device: &DeviceArc,
    map: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
    ignore_background: bool,
) -> Result<ArrayPtr> {
    apply_touching_neighbors_operation(
        device,
        map,
        labels,
        dst,
        radius,
        ignore_background,
        tier1::minimum_of_touching_neighbors,
    )
}

/// Mirrors CLIc's `maximum_of_touching_neighbors_map_func`.
pub fn maximum_of_touching_neighbors_map(
    device: &DeviceArc,
    map: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
    ignore_background: bool,
) -> Result<ArrayPtr> {
    apply_touching_neighbors_operation(
        device,
        map,
        labels,
        dst,
        radius,
        ignore_background,
        tier1::maximum_of_touching_neighbors,
    )
}

/// Mirrors CLIc's `standard_deviation_of_touching_neighbors_map_func`.
pub fn standard_deviation_of_touching_neighbors_map(
    device: &DeviceArc,
    map: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
    ignore_background: bool,
) -> Result<ArrayPtr> {
    apply_touching_neighbors_operation(
        device,
        map,
        labels,
        dst,
        radius,
        ignore_background,
        tier1::standard_deviation_of_touching_neighbors,
    )
}

/// Mirrors CLIc's `mode_of_touching_neighbors_map_func`.
pub fn mode_of_touching_neighbors_map(
    device: &DeviceArc,
    map: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
    ignore_background: bool,
) -> Result<ArrayPtr> {
    apply_touching_neighbors_operation(
        device,
        map,
        labels,
        dst,
        radius,
        ignore_background,
        tier1::mode_of_touching_neighbors,
    )
}

/// Mirrors CLIc's `median_of_touching_neighbors_map_func`.
pub fn median_of_touching_neighbors_map(
    device: &DeviceArc,
    map: &ArrayPtr,
    labels: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
    ignore_background: bool,
) -> Result<ArrayPtr> {
    apply_touching_neighbors_operation(
        device,
        map,
        labels,
        dst,
        radius,
        ignore_background,
        tier1::median_of_touching_neighbors,
    )
}
