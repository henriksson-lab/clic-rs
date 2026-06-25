use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::types::DType;

/// Return the sum of all pixel values.
pub fn sum_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let dst = tier0::create_one(src, None, DType::Float, device)?;
    let mut tmp = src.clone();

    let project_if_needed =
        |tmp: &mut ArrayPtr,
         projection_func: fn(&DeviceArc, &ArrayPtr, Option<ArrayPtr>) -> Result<ArrayPtr>,
         dimension: usize|
         -> Result<()> {
            if dimension > 1 {
                *tmp = projection_func(device, tmp, None)?;
            }
            Ok(())
        };

    let dimension = tmp.lock().unwrap().depth();
    project_if_needed(&mut tmp, tier1::sum_z_projection, dimension)?;
    let dimension = tmp.lock().unwrap().height();
    project_if_needed(&mut tmp, tier1::sum_y_projection, dimension)?;
    tier1::sum_x_projection(device, &tmp, Some(dst.clone()))?;

    let mut v = [0.0_f32; 1];
    dst.lock().unwrap().read_to(&mut v)?;
    Ok(v[0])
}
