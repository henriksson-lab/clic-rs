use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::{tier1, tier3};

pub fn label_bounding_box(device: &DeviceArc, src: &ArrayPtr, label_id: i32) -> Result<Vec<f32>> {
    let binary = tier1::equal_constant(device, src, None, label_id as f32)?;
    tier3::bounding_box(device, &binary)
}
