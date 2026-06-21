use crate::array::{pull, Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::types::{MType, LABEL};

fn exclude_axis(device: &DeviceArc, src: &ArrayPtr, label_map: &ArrayPtr, axis: i32) -> Result<()> {
    let (width, height, depth) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.depth())
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(label_map.clone())),
    ];
    let constants = vec![("EXCLUDE_AXIS", ConstantValue::Int(axis))];
    execute(
        device,
        (
            "exclude_on_edges",
            include_str!("../../kernels/exclude_on_edges.cl"),
        ),
        &params,
        [width, height, depth],
        [1, 1, 1],
        &constants,
    )
}

pub fn remove_labels_on_edges(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    exclude_x: bool,
    exclude_y: bool,
    exclude_z: bool,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    let num_labels = tier2::maximum_of_all_pixels(device, src)? as usize;
    let label_map = Array::create(num_labels + 1, 1, 1, 1, LABEL, MType::Buffer, device)?;
    tier1::set_ramp_x(device, &label_map)?;

    let (width, height, depth) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.depth())
    };
    if exclude_x && width > 1 {
        exclude_axis(device, src, &label_map, 0)?;
    }
    if exclude_y && height > 1 {
        exclude_axis(device, src, &label_map, 1)?;
    }
    if exclude_z && depth > 1 {
        exclude_axis(device, src, &label_map, 2)?;
    }

    let mut label_map_vector: Vec<u32> = pull(&label_map)?;
    let mut count = 1_u32;
    for value in &mut label_map_vector {
        if *value > 0 {
            *value = count;
            count += 1;
        }
    }
    label_map
        .lock()
        .unwrap()
        .write_from_typed(&label_map_vector)?;
    tier1::replace_values(device, src, &label_map, Some(dst.clone()))?;
    Ok(dst)
}

pub fn exclude_labels_on_edges(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    exclude_x: bool,
    exclude_y: bool,
    exclude_z: bool,
) -> Result<ArrayPtr> {
    remove_labels_on_edges(device, src, dst, exclude_x, exclude_y, exclude_z)
}
