use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::types::{DType, MType, LABEL};

pub fn extend_labeling_via_voronoi(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    let flip = Array::create_from_array(&dst)?;
    let flop = Array::create_from_array(&dst)?;
    tier1::copy(device, src, Some(flip.clone()))?;
    let flag = Array::create(1, 1, 1, 1, DType::Int32, MType::Buffer, device)?;
    flag.lock().unwrap().fill(0.0)?;

    let mut flag_value = 1_i32;
    let mut iteration_count = 0;
    while flag_value > 0 {
        if iteration_count % 2 == 0 {
            tier1::onlyzero_overwrite_maximum(device, &flip, &flag, Some(flop.clone()), "box")?;
        } else {
            tier1::onlyzero_overwrite_maximum(device, &flop, &flag, Some(flip.clone()), "sphere")?;
        }
        let mut flag_host = [0_i32; 1];
        flag.lock().unwrap().read_to(&mut flag_host)?;
        flag_value = flag_host[0];
        flag.lock().unwrap().fill(0.0)?;
        iteration_count += 1;
    }

    if iteration_count % 2 == 0 {
        flip.lock().unwrap().copy_to(&dst)?;
    } else {
        flop.lock().unwrap().copy_to(&dst)?;
    }
    Ok(dst)
}
