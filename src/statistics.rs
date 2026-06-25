use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier1;
use crate::tier2;
use crate::types::{DType, MType};

pub type StatisticsMap = std::collections::HashMap<String, Vec<f32>>;

pub fn _statistics_per_label(
    device: &DeviceArc,
    label: &ArrayPtr,
    intensity: &ArrayPtr,
    nb_labels: i32,
) -> Result<ArrayPtr> {
    let min_value = f32::MIN;
    let max_value = f32::MAX;
    let (height, depth) = {
        let label = label.lock().unwrap();
        (label.height(), label.depth())
    };

    let cumulative_stats_per_label = Array::create(
        nb_labels as usize,
        height,
        16,
        3,
        DType::Float,
        MType::Buffer,
        device,
    )?;
    cumulative_stats_per_label.lock().unwrap().fill(0.0)?;

    for i in 8..=15 {
        let value = if i % 2 == 0 { max_value } else { min_value };
        tier1::set_plane(device, &cumulative_stats_per_label, i, value)?;
    }

    let kernel = (
        "statistics_per_label",
        include_str!("../kernels/statistics_per_label.cl"),
    );
    let range = [1, height, 1];
    let mut params = vec![
        ("src_label", ParameterValue::Array(label.clone())),
        ("src_image", ParameterValue::Array(intensity.clone())),
        (
            "dst",
            ParameterValue::Array(cumulative_stats_per_label.clone()),
        ),
        ("sum_background", ParameterValue::Int(0)),
        ("z", ParameterValue::Int(0)),
    ];
    for z in 0..depth {
        let it = params
            .iter_mut()
            .find(|param| param.0 == "z")
            .expect("z parameter exists");
        it.1 = ParameterValue::Int(z as i32);
        execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    }

    Ok(cumulative_stats_per_label)
}

pub fn _std_per_label(
    device: &DeviceArc,
    statistics: &ArrayPtr,
    label: &ArrayPtr,
    intensity: &ArrayPtr,
    nb_labels: i32,
) -> Result<ArrayPtr> {
    let (height, depth) = {
        let label = label.lock().unwrap();
        (label.height(), label.depth())
    };

    let label_statistics_stack = Array::create(
        nb_labels as usize,
        height,
        6,
        3,
        DType::Float,
        MType::Buffer,
        device,
    )?;
    label_statistics_stack.lock().unwrap().fill(0.0)?;

    let kernel_std = (
        "standard_deviation_per_label",
        include_str!("../kernels/standard_deviation_per_label.cl"),
    );
    let range_std = [1, height, 1];
    let mut params_std = vec![
        ("src_statistics", ParameterValue::Array(statistics.clone())),
        ("src_label", ParameterValue::Array(label.clone())),
        ("src_image", ParameterValue::Array(intensity.clone())),
        ("dst", ParameterValue::Array(label_statistics_stack.clone())),
        ("sum_background", ParameterValue::Int(0)),
        ("z", ParameterValue::Int(0)),
    ];
    for z in 0..depth {
        let it = params_std
            .iter_mut()
            .find(|param| param.0 == "z")
            .expect("z parameter exists");
        it.1 = ParameterValue::Int(z as i32);
        execute(device, kernel_std, &params_std, range_std, [0, 0, 0], &[])?;
    }

    Ok(label_statistics_stack)
}

pub fn compute_statistics_per_labels(
    device: &DeviceArc,
    label: &ArrayPtr,
    intensity: &ArrayPtr,
) -> Result<StatisticsMap> {
    // initialize variables, output, and constants
    let offset = 1usize;
    let nb_labels = tier2::maximum_of_all_pixels(device, label)? as usize + offset;
    let nb_measurements = nb_labels - offset;
    let origin = [0, 0, 0];
    let region = [nb_measurements, 1, 1];

    // create output map and temp vector in GPU and CPU
    let mut region_props = StatisticsMap::new();
    region_props.reserve(37);
    let result_device_vector = Array::create(
        nb_measurements,
        1,
        1,
        1,
        DType::Float,
        MType::Buffer,
        device,
    )?;

    // compute statistics per label and collect slice-by-slice measurements in single planes
    let cumulative_stats_per_label =
        _statistics_per_label(device, label, intensity, nb_labels as i32)?;
    let sum_per_label = tier1::sum_y_projection(device, &cumulative_stats_per_label, None)?;
    let min_per_label = tier1::minimum_y_projection(device, &cumulative_stats_per_label, None)?;
    let max_per_label = tier1::maximum_y_projection(device, &cumulative_stats_per_label, None)?;

    let label_statistics_image =
        Array::create(nb_labels, 8, 1, 2, DType::Float, MType::Buffer, device)?;
    let sum_over_dimensions = Array::create_from_array(&result_device_vector)?;
    let avg_over_dimensions = Array::create_from_array(&result_device_vector)?;

    // 0 labels
    let labels_list = (offset..=nb_measurements).map(|v| v as f32).collect();
    region_props.insert("label".to_string(), labels_list);

    // 1-6 bbox x, y, z, width, height, depth
    /*
      # 10. min_x: minimum x coordinate of the label (in the given column)
      # 11. max_x: maximum x coordinate of the label
      # 12. min_y: minimum y coordinate of the label
      # 13. max_y: maximum y coordinate of the label
      # 14. min_z: minimum z coordinate of the label
      # 15. max_z: maximum z coordinate of the label
    */
    // min and max bbox x, y, z
    let mut bbox_min_x = vec![0.0f32; nb_measurements];
    min_per_label
        .lock()
        .unwrap()
        .read_to_region(&mut bbox_min_x, region, [offset, 10, 0])?;
    region_props.insert("bbox_min_x".to_string(), bbox_min_x);
    let mut bbox_min_y = vec![0.0f32; nb_measurements];
    min_per_label
        .lock()
        .unwrap()
        .read_to_region(&mut bbox_min_y, region, [offset, 12, 0])?;
    region_props.insert("bbox_min_y".to_string(), bbox_min_y);
    let mut bbox_min_z = vec![0.0f32; nb_measurements];
    min_per_label
        .lock()
        .unwrap()
        .read_to_region(&mut bbox_min_z, region, [offset, 14, 0])?;
    region_props.insert("bbox_min_z".to_string(), bbox_min_z);
    let mut bbox_max_x = vec![0.0f32; nb_measurements];
    max_per_label
        .lock()
        .unwrap()
        .read_to_region(&mut bbox_max_x, region, [offset, 11, 0])?;
    region_props.insert("bbox_max_x".to_string(), bbox_max_x);
    let mut bbox_max_y = vec![0.0f32; nb_measurements];
    max_per_label
        .lock()
        .unwrap()
        .read_to_region(&mut bbox_max_y, region, [offset, 13, 0])?;
    region_props.insert("bbox_max_y".to_string(), bbox_max_y);
    let mut bbox_max_z = vec![0.0f32; nb_measurements];
    max_per_label
        .lock()
        .unwrap()
        .read_to_region(&mut bbox_max_z, region, [offset, 15, 0])?;
    region_props.insert("bbox_max_z".to_string(), bbox_max_z);

    // bbox width, height, depth
    let mut bbox_width = vec![0.0f32; nb_measurements];
    let mut bbox_height = vec![0.0f32; nb_measurements];
    let mut bbox_depth = vec![0.0f32; nb_measurements];
    for i in 0..nb_measurements {
        bbox_width[i] = region_props["bbox_max_x"][i] - region_props["bbox_min_x"][i] + 1.0;
        bbox_height[i] = region_props["bbox_max_y"][i] - region_props["bbox_min_y"][i] + 1.0;
        bbox_depth[i] = region_props["bbox_max_z"][i] - region_props["bbox_min_z"][i] + 1.0;
    }
    region_props.insert("bbox_width".to_string(), bbox_width);
    region_props.insert("bbox_height".to_string(), bbox_height);
    region_props.insert("bbox_depth".to_string(), bbox_depth);

    /*
      # 0. sum_x: all x-coordinates summed per label (and column)
      # 1. sum_y: all y-coordinates summed per label
      # 2. sum_z: all z-coordinates summed per label
      # 3. sum: number of pixels per label (sum of a binary image representing the label)
      # 4. sum_intensity_x: intensity (of the intensity image) times x-coordinate summed per label
      # 5. sum_intensity_y: intensity times y-coordinate summed per label
      # 6. sum_intensity_z: intensity times z-coordinate summed per label
      # 7. sum_intensity: sum intensity (a.k.a. total intensity) of the label
      # 8. min_intensity: minimum intensity of the label
      # 9. max_intensity: maximum intensity of the label
    */
    // Area, minimum, maximum, sum, and mean intensity
    let mut area = vec![0.0f32; nb_measurements];
    sum_per_label.lock().unwrap().copy_to_region(
        &sum_over_dimensions,
        region,
        [offset, 3, 0],
        origin,
    )?;
    sum_over_dimensions.lock().unwrap().read_to(&mut area)?;
    region_props.insert("area".to_string(), area);
    let mut min_intensity = vec![0.0f32; nb_measurements];
    min_per_label
        .lock()
        .unwrap()
        .read_to_region(&mut min_intensity, region, [offset, 8, 0])?;
    region_props.insert("min_intensity".to_string(), min_intensity);
    let mut max_intensity = vec![0.0f32; nb_measurements];
    max_per_label
        .lock()
        .unwrap()
        .read_to_region(&mut max_intensity, region, [offset, 9, 0])?;
    region_props.insert("max_intensity".to_string(), max_intensity);
    let mut sum_intensity = vec![0.0f32; nb_measurements];
    sum_per_label.lock().unwrap().copy_to_region(
        &result_device_vector,
        region,
        [offset, 7, 0],
        origin,
    )?;
    result_device_vector
        .lock()
        .unwrap()
        .read_to(&mut sum_intensity)?;
    region_props.insert("sum_intensity".to_string(), sum_intensity);
    let mut mean_intensity = vec![0.0f32; nb_measurements];
    tier1::paste(
        device,
        &sum_over_dimensions,
        Some(label_statistics_image.clone()),
        offset as i32,
        7,
        0,
    )?;
    tier1::divide_images(
        device,
        &result_device_vector,
        &sum_over_dimensions,
        Some(avg_over_dimensions.clone()),
    )?;
    tier1::paste(
        device,
        &avg_over_dimensions,
        Some(label_statistics_image.clone()),
        offset as i32,
        6,
        0,
    )?;
    avg_over_dimensions
        .lock()
        .unwrap()
        .read_to(&mut mean_intensity)?;
    region_props.insert("mean_intensity".to_string(), mean_intensity);

    // Sum intensity times x, y, z and mass center
    let dim_names = ["x", "y", "z"];
    sum_per_label.lock().unwrap().copy_to_region(
        &result_device_vector,
        region,
        [offset, 4 + 3, 0],
        origin,
    )?;

    for dim in 0..3 {
        let mut sum_intensity_times_ = vec![0.0f32; nb_measurements];
        sum_per_label.lock().unwrap().copy_to_region(
            &sum_over_dimensions,
            region,
            [offset, 4 + dim, 0],
            origin,
        )?;
        sum_over_dimensions
            .lock()
            .unwrap()
            .read_to(&mut sum_intensity_times_)?;
        region_props.insert(
            "sum_intensity_times_".to_string() + dim_names[dim],
            sum_intensity_times_,
        );
        let mut mass_center_ = vec![0.0f32; nb_measurements];
        tier1::divide_images(
            device,
            &sum_over_dimensions,
            &result_device_vector,
            Some(avg_over_dimensions.clone()),
        )?;
        avg_over_dimensions
            .lock()
            .unwrap()
            .read_to(&mut mass_center_)?;
        region_props.insert("mass_center_".to_string() + dim_names[dim], mass_center_);
        tier1::paste(
            device,
            &avg_over_dimensions,
            Some(label_statistics_image.clone()),
            offset as i32,
            (3 + dim) as i32,
            0,
        )?;
    }

    // Sum x, y, z and centroid
    sum_per_label.lock().unwrap().copy_to_region(
        &result_device_vector,
        region,
        [offset, 3, 0],
        origin,
    )?;
    for dim in 0..3 {
        let mut sum_ = vec![0.0f32; nb_measurements];
        sum_per_label.lock().unwrap().copy_to_region(
            &sum_over_dimensions,
            region,
            [offset, dim, 0],
            origin,
        )?;
        sum_over_dimensions.lock().unwrap().read_to(&mut sum_)?;
        region_props.insert("sum_".to_string() + dim_names[dim], sum_);
        let mut centroid_ = vec![0.0f32; nb_measurements];
        tier1::divide_images(
            device,
            &sum_over_dimensions,
            &result_device_vector,
            Some(avg_over_dimensions.clone()),
        )?;
        avg_over_dimensions
            .lock()
            .unwrap()
            .read_to(&mut centroid_)?;
        region_props.insert("centroid_".to_string() + dim_names[dim], centroid_);
        tier1::paste(
            device,
            &avg_over_dimensions,
            Some(label_statistics_image.clone()),
            offset as i32,
            dim as i32,
            0,
        )?;
    }

    // Second part: determine parameters which depend on other parameters
    let label_statistics_stack = _std_per_label(
        device,
        &label_statistics_image,
        label,
        intensity,
        nb_labels as i32,
    )?;
    let sum_statistics = tier1::sum_y_projection(device, &label_statistics_stack, None)?;
    let max_statistics = tier1::maximum_y_projection(device, &label_statistics_stack, None)?;
    sum_per_label.lock().unwrap().copy_to_region(
        &result_device_vector,
        region,
        [offset, 3, 0],
        origin,
    )?;

    // Sum and mean distance to centroid
    let mut sum_distance_to_centroid = vec![0.0f32; nb_measurements];
    sum_statistics.lock().unwrap().copy_to_region(
        &sum_over_dimensions,
        region,
        [offset, 0, 0],
        origin,
    )?;
    sum_over_dimensions
        .lock()
        .unwrap()
        .read_to(&mut sum_distance_to_centroid)?;
    region_props.insert(
        "sum_distance_to_centroid".to_string(),
        sum_distance_to_centroid,
    );
    let mut mean_distance_to_centroid = vec![0.0f32; nb_measurements];
    tier1::divide_images(
        device,
        &sum_over_dimensions,
        &result_device_vector,
        Some(avg_over_dimensions.clone()),
    )?;
    avg_over_dimensions
        .lock()
        .unwrap()
        .read_to(&mut mean_distance_to_centroid)?;
    region_props.insert(
        "mean_distance_to_centroid".to_string(),
        mean_distance_to_centroid,
    );
    // Sum and mean distance to center of mass
    let mut sum_distance_to_mass_center = vec![0.0f32; nb_measurements];
    sum_statistics.lock().unwrap().copy_to_region(
        &sum_over_dimensions,
        region,
        [offset, 1, 0],
        origin,
    )?;
    sum_over_dimensions
        .lock()
        .unwrap()
        .read_to(&mut sum_distance_to_mass_center)?;
    region_props.insert(
        "sum_distance_to_mass_center".to_string(),
        sum_distance_to_mass_center,
    );
    let mut mean_distance_to_mass_center = vec![0.0f32; nb_measurements];
    tier1::divide_images(
        device,
        &sum_over_dimensions,
        &result_device_vector,
        Some(avg_over_dimensions.clone()),
    )?;
    avg_over_dimensions
        .lock()
        .unwrap()
        .read_to(&mut mean_distance_to_mass_center)?;
    region_props.insert(
        "mean_distance_to_mass_center".to_string(),
        mean_distance_to_mass_center,
    );
    // Standard deviation intensity
    let mut standard_deviation_intensity = vec![0.0f32; nb_measurements];
    sum_statistics.lock().unwrap().copy_to_region(
        &sum_over_dimensions,
        region,
        [offset, 2, 0],
        origin,
    )?;
    tier1::power(
        device,
        &sum_over_dimensions,
        Some(result_device_vector.clone()),
        0.5,
    )?;
    result_device_vector
        .lock()
        .unwrap()
        .read_to(&mut standard_deviation_intensity)?;
    region_props.insert(
        "standard_deviation_intensity".to_string(),
        standard_deviation_intensity,
    );
    // Max distance to centroid and center of mass
    let mut max_distance_to_centroid = vec![0.0f32; nb_measurements];
    max_statistics.lock().unwrap().read_to_region(
        &mut max_distance_to_centroid,
        region,
        [offset, 4, 0],
    )?;
    region_props.insert(
        "max_distance_to_centroid".to_string(),
        max_distance_to_centroid,
    );
    let mut max_distance_to_mass_center = vec![0.0f32; nb_measurements];
    max_statistics.lock().unwrap().read_to_region(
        &mut max_distance_to_mass_center,
        region,
        [offset, 5, 0],
    )?;
    region_props.insert(
        "max_distance_to_mass_center".to_string(),
        max_distance_to_mass_center,
    );

    // Calculate distance ratios
    let mut mean_max_distance_to_centroid_ratio = vec![0.0f32; nb_measurements];
    let mut mean_max_distance_to_mass_center_ratio = vec![0.0f32; nb_measurements];
    for i in 0..nb_measurements {
        mean_max_distance_to_centroid_ratio[i] = region_props["max_distance_to_centroid"][i]
            / region_props["mean_distance_to_centroid"][i];
        mean_max_distance_to_mass_center_ratio[i] = region_props["max_distance_to_mass_center"][i]
            / region_props["mean_distance_to_mass_center"][i];
    }
    region_props.insert(
        "mean_max_distance_to_centroid_ratio".to_string(),
        mean_max_distance_to_centroid_ratio,
    );
    region_props.insert(
        "mean_max_distance_to_mass_center_ratio".to_string(),
        mean_max_distance_to_mass_center_ratio,
    );

    Ok(region_props)
}
