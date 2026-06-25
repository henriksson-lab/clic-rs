use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

/// Return the position of the first maximum value as `[x, y, z]`.
///
/// Mirrors CLIc's `maximum_position_func`.
pub fn maximum_position(device: &DeviceArc, src: &ArrayPtr) -> Result<Vec<f32>> {
    let mut z_coord = 0usize;
    let mut y_coord = 0usize;
    let x_coord;
    let mut coord = vec![0.0f32; 3];

    let mut temp = src.clone();
    let mut pos_z = None;
    let mut pos_y = None;

    if src.lock().unwrap().depth() > 1 {
        let pos = tier1::z_position_of_maximum_z_projection(device, &temp, None)?;
        temp = tier1::maximum_z_projection(device, &temp, None)?;
        pos_z = Some(pos);
    }
    if src.lock().unwrap().height() > 1 {
        let pos = tier1::y_position_of_maximum_y_projection(device, &temp, None)?;
        temp = tier1::maximum_y_projection(device, &temp, None)?;
        pos_y = Some(pos);
    }

    let pos_x = tier1::x_position_of_maximum_x_projection(device, &temp, None)?;
    temp = tier1::maximum_x_projection(device, &temp, None)?;

    let mut value = [0u32; 1];
    pos_x.lock().unwrap().read_to_at(&mut value, 0, 0, 0)?;
    x_coord = value[0] as usize;
    coord[0] = x_coord as f32;

    if let Some(pos_y) = pos_y {
        pos_y
            .lock()
            .unwrap()
            .read_to_at(&mut value, x_coord, 0, 0)?;
        y_coord = value[0] as usize;
        coord[1] = y_coord as f32;
    }
    if let Some(pos_z) = pos_z {
        pos_z
            .lock()
            .unwrap()
            .read_to_at(&mut value, x_coord, y_coord, 0)?;
        z_coord = value[0] as usize;
    }
    coord[2] = z_coord as f32;

    let _ = temp;

    Ok(coord)
}
