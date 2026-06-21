use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::types::DType;

/// Depth projection using local variance maxima to determine the best focus plane.
pub fn extended_depth_of_focus_variance_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius_x: f32,
    radius_y: f32,
    sigma: f32,
) -> Result<ArrayPtr> {
    let variance = tier1::variance_filter(device, src, None, radius_x, radius_y, 0.0, "sphere")?;
    let temp = tier1::gaussian_blur(device, &variance, None, sigma, sigma, 0.0)?;
    let altitude = tier1::z_position_of_maximum_z_projection(device, &temp, None)?;
    tier1::z_position_projection(device, src, &altitude, dst)
}

/// Depth projection using local Sobel gradient magnitude maxima to determine the best focus plane.
pub fn extended_depth_of_focus_sobel_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    sigma: f32,
) -> Result<ArrayPtr> {
    let temp = tier0::create_like(src, None, DType::Unknown, device)?;
    let temp_2d = tier0::create_xy(src, None, DType::Unknown, device)?;
    let depth = {
        let src = src.lock().unwrap();
        src.depth()
    };

    for z in 0..depth {
        tier1::copy_slice(device, src, Some(temp_2d.clone()), z as i32)?;
        let sobel = tier1::sobel(device, &temp_2d, None)?;
        let blurred = tier1::gaussian_blur(device, &sobel, None, sigma, sigma, 0.0)?;
        tier1::copy_slice(device, &blurred, Some(temp.clone()), z as i32)?;
    }

    let altitude = tier1::z_position_of_maximum_z_projection(device, &temp, None)?;
    tier1::z_position_projection(device, src, &altitude, dst)
}
