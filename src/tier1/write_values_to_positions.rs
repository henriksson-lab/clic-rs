use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::utils::shape_to_dimension;

use super::projections::maximum_x_projection;

/// Write point-list values into their encoded positions.
///
/// Mirrors CLIc's `write_values_to_positions_func`.
pub fn write_values_to_positions(
    device: &DeviceArc,
    list: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    // TODO: check list shape
    // if (list->dim() != 2)
    // {
    //   throw std::runtime_error(
    //     "The Position list is expected to be 2D, where rows are positions (x,y,z) and values v.");
    // }
    let mut dst = dst;
    if dst.is_none() {
        let (height, mtype, list_device) = {
            let list = list.lock().unwrap();
            (list.height(), list.mtype(), list.device().clone())
        };
        // flatten the coords to get the max position value in x,y,z
        // as well as the number of rows (2->1D, 3->2D, 4->3D)
        let temp = Array::create(
            1,
            height,
            1,
            2,
            crate::types::DType::Int32,
            mtype,
            &list_device,
        )?;
        maximum_x_projection(device, list, Some(temp.clone()))?;
        let mut max_position = vec![0i32; temp.lock().unwrap().size()];
        temp.lock().unwrap().read_to(&mut max_position)?;

        let max_pos_x = (max_position[0] + 1) as usize;
        let max_pos_y = if height > 2 {
            (max_position[1] + 1) as usize
        } else {
            1
        };
        let max_pos_z = if height > 3 {
            (max_position[2] + 1) as usize
        } else {
            1
        };

        let (dtype, mtype, list_device) = {
            let list = list.lock().unwrap();
            (list.dtype(), list.mtype(), list.device().clone())
        };
        let dim = shape_to_dimension(max_pos_x, max_pos_y, max_pos_z);
        dst = Some(Array::create(
            max_pos_x,
            max_pos_y,
            max_pos_z,
            dim,
            dtype,
            mtype,
            &list_device,
        )?);
        dst.as_ref().unwrap().lock().unwrap().fill(0.0)?;
    }
    let dst = dst.unwrap();
    let kernel = (
        "write_values_to_positions",
        include_str!("../../kernels/write_values_to_positions.cl"),
    );
    let params = vec![
        ("src", ParameterValue::Array(list.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let list = list.lock().unwrap();
        [list.width(), 1, 1]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
