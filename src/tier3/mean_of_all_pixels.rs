use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier2;

/// Return the mean pixel value of the entire array.
pub fn mean_of_all_pixels(device: &DeviceArc, src: &ArrayPtr) -> Result<f32> {
    let sum = tier2::sum_of_all_pixels(device, src)?;
    let n = src.lock().unwrap().size();
    Ok(sum / n as f32)
}
