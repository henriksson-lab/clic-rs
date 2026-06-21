use crate::array::{pull, Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::utils::shape_to_dimension;

use super::projections::maximum_x_projection;

fn create_dst_from_position_list(device: &DeviceArc, list: &ArrayPtr) -> Result<ArrayPtr> {
    let list_height = list.lock().unwrap().height();
    let temp = Array::create(
        1,
        list_height,
        1,
        2,
        crate::types::DType::Int32,
        {
            let list = list.lock().unwrap();
            list.mtype()
        },
        device,
    )?;
    maximum_x_projection(device, list, Some(temp.clone()))?;
    let max_position = pull::<i32>(&temp)?;

    let max_pos_x = max_position[0] as usize + 1;
    let max_pos_y = if list_height > 2 {
        max_position[1] as usize + 1
    } else {
        1
    };
    let max_pos_z = if list_height > 3 {
        max_position[2] as usize + 1
    } else {
        1
    };

    let list = list.lock().unwrap();
    let dim = shape_to_dimension(max_pos_x, max_pos_y, max_pos_z);
    let dst = Array::create(
        max_pos_x,
        max_pos_y,
        max_pos_z,
        dim,
        list.dtype(),
        list.mtype(),
        device,
    )?;
    dst.lock().unwrap().fill(0.0)?;
    Ok(dst)
}

/// Write point-list values into their encoded positions.
///
/// Mirrors CLIc's `write_values_to_positions_func`.
pub fn write_values_to_positions(
    device: &DeviceArc,
    list: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = match dst {
        Some(dst) => dst,
        None => create_dst_from_position_list(device, list)?,
    };
    let width = list.lock().unwrap().width();
    let params = vec![
        ("src", ParameterValue::Array(list.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        (
            "write_values_to_positions",
            include_str!("../../kernels/write_values_to_positions.cl"),
        ),
        &params,
        [width, 1, 1],
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
