use crate::array::{pull, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

fn read_index(arr: &ArrayPtr, x: usize, y: usize) -> Result<usize> {
    let width = arr.lock().unwrap().width();
    let values: Vec<u32> = pull(arr)?;
    Ok(values[y * width + x] as usize)
}

/// Return the position of the first minimum value as `[x, y, z]`.
///
/// Mirrors CLIc's `minimum_position_func`.
pub fn minimum_position(device: &DeviceArc, src: &ArrayPtr) -> Result<Vec<f32>> {
    let (height, depth) = {
        let l = src.lock().unwrap();
        (l.height(), l.depth())
    };

    let mut y_coord = 0usize;
    let mut z_coord = 0usize;
    let mut temp = src.clone();
    let mut pos_z = None;
    let mut pos_y = None;

    if depth > 1 {
        let positions = tier1::z_position_of_minimum_z_projection(device, &temp, None)?;
        temp = tier1::minimum_z_projection(device, &temp, None)?;
        pos_z = Some(positions);
    }
    if height > 1 {
        let positions = tier1::y_position_of_minimum_y_projection(device, &temp, None)?;
        temp = tier1::minimum_y_projection(device, &temp, None)?;
        pos_y = Some(positions);
    }

    let pos_x = tier1::x_position_of_minimum_x_projection(device, &temp, None)?;
    let x_coord = read_index(&pos_x, 0, 0)?;
    if let Some(pos_y) = pos_y {
        y_coord = read_index(&pos_y, x_coord, 0)?;
    }
    if let Some(pos_z) = pos_z {
        z_coord = read_index(&pos_z, x_coord, y_coord)?;
    }

    Ok(vec![x_coord as f32, y_coord as f32, z_coord as f32])
}
