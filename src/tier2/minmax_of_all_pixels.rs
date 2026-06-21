use crate::array::{pull, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;

/// Return the maximum pixel value across the entire array.
pub fn maximum_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let (_, h, d) = {
        let l = src.lock().unwrap();
        (l.width(), l.height(), l.depth())
    };
    let mut tmp = src.clone();
    if d > 1 {
        tmp = tier1::maximum_z_projection(device, &tmp, None)?;
    }
    if h > 1 {
        tmp = tier1::maximum_y_projection(device, &tmp, None)?;
    }
    let dst = tier0::create_one(device)?;
    tier1::maximum_x_projection(device, &tmp, Some(dst.clone()))?;
    let v: Vec<f32> = pull(&dst)?;
    Ok(v[0])
}

/// Return the minimum pixel value across the entire array.
pub fn minimum_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let (_, h, d) = {
        let l = src.lock().unwrap();
        (l.width(), l.height(), l.depth())
    };
    let mut tmp = src.clone();
    if d > 1 {
        tmp = tier1::minimum_z_projection(device, &tmp, None)?;
    }
    if h > 1 {
        tmp = tier1::minimum_y_projection(device, &tmp, None)?;
    }
    let dst = tier0::create_one(device)?;
    tier1::minimum_x_projection(device, &tmp, Some(dst.clone()))?;
    let v: Vec<f32> = pull(&dst)?;
    Ok(v[0])
}
