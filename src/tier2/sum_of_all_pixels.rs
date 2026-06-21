use crate::array::{pull, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;

/// Return the sum of all pixel values.
pub fn sum_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let (_, h, d) = {
        let l = src.lock().unwrap();
        (l.width(), l.height(), l.depth())
    };
    let mut tmp = src.clone();
    if d > 1 {
        tmp = tier1::sum_z_projection(device, &tmp, None)?;
    }
    if h > 1 {
        tmp = tier1::sum_y_projection(device, &tmp, None)?;
    }
    let dst = tier0::create_one(device)?;
    tier1::sum_x_projection(device, &tmp, Some(dst.clone()))?;
    let v: Vec<f32> = pull(&dst)?;
    Ok(v[0])
}
