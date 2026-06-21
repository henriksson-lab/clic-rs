use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::{tier1, tier2};

pub fn bounding_box(device: &DeviceArc, src: &ArrayPtr) -> Result<Vec<f32>> {
    let temp_x = tier1::multiply_image_and_position(device, src, None, 0)?;
    let max_x = tier2::maximum_of_all_pixels(device, &temp_x)?;
    let min_x = tier2::minimum_of_masked_pixels(device, &temp_x, src)?;

    let temp_y = tier1::multiply_image_and_position(device, src, None, 1)?;
    let max_y = tier2::maximum_of_all_pixels(device, &temp_y)?;
    let min_y = tier2::minimum_of_masked_pixels(device, &temp_y, src)?;

    let (min_z, max_z) = if src.lock().unwrap().depth() > 1 {
        let temp_z = tier1::multiply_image_and_position(device, src, None, 2)?;
        (
            tier2::minimum_of_masked_pixels(device, &temp_z, src)?,
            tier2::maximum_of_all_pixels(device, &temp_z)?,
        )
    } else {
        (0.0, 0.0)
    };

    Ok(vec![min_x, min_y, min_z, max_x, max_y, max_z])
}
