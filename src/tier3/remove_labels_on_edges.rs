use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::types::{MType, LABEL};

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
    let src_device = src.lock().unwrap().device().clone();
    let label_map = Array::create(num_labels + 1, 1, 1, 1, LABEL, MType::Buffer, &src_device)?;
    tier1::set_ramp_x(device, &label_map)?;

    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(label_map.clone())),
    ];
    let execute_if_needed = |exclude: bool, dimension: usize, axis: i32| -> Result<()> {
        if exclude && dimension > 1 {
            let kernel = (
                "exclude_on_edges",
                include_str!("../../kernels/exclude_on_edges.cl"),
            );
            let range = {
                let src = src.lock().unwrap();
                [
                    if dimension == 0 { 1 } else { src.width() },
                    if dimension == 1 { 1 } else { src.height() },
                    if dimension == 2 { 1 } else { src.depth() },
                ]
            };
            let local = [1, 1, 1];
            let constants = vec![("EXCLUDE_AXIS", ConstantValue::Int(axis))];
            execute(device, kernel, &params, range, local, &constants)?;
        }
        Ok(())
    };

    execute_if_needed(exclude_x, src.lock().unwrap().width(), 0)?;
    execute_if_needed(exclude_y, src.lock().unwrap().height(), 1)?;
    execute_if_needed(exclude_z, src.lock().unwrap().depth(), 2)?;

    let mut label_map_vector = vec![0_u32; num_labels + 1];
    label_map.lock().unwrap().read_to(&mut label_map_vector)?;
    let mut count = 1_u32;
    for i in &mut label_map_vector {
        if *i > 0 {
            *i = count;
            count += 1;
        }
    }
    label_map.lock().unwrap().write_from(&label_map_vector)?;
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
