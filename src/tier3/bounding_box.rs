use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::{tier1, tier2};

pub fn bounding_box(device: &DeviceArc, src: &ArrayPtr) -> Result<Vec<f32>> {
    let min_x;
    let min_y;
    let mut min_z = 0.0;
    let max_x;
    let max_y;
    let mut max_z = 0.0;

    let mut temp = tier1::multiply_image_and_position(device, src, None, 0)?;
    max_x = tier2::maximum_of_all_pixels(device, &temp)?;
    min_x = tier2::minimum_of_masked_pixels(device, &temp, src)?;

    temp = tier1::multiply_image_and_position(device, src, None, 1)?;
    max_y = tier2::maximum_of_all_pixels(device, &temp)?;
    min_y = tier2::minimum_of_masked_pixels(device, &temp, src)?;

    if src.lock().unwrap().depth() > 1 {
        temp = tier1::multiply_image_and_position(device, src, None, 2)?;
        max_z = tier2::maximum_of_all_pixels(device, &temp)?;
        min_z = tier2::minimum_of_masked_pixels(device, &temp, src)?;
    }

    Ok(vec![min_x, min_y, min_z, max_x, max_y, max_z])
}
