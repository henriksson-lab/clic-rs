use crate::array::{pull, Array, ArrayPtr};
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
    let flip = tier1::copy(device, src, None)?;
    let flop = tier0::create_like(&dst, None, DType::Unknown, device)?;
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
        let flag_host: Vec<i32> = pull(&flag)?;
        flag_value = flag_host[0];
        flag.lock().unwrap().fill(0.0)?;
        iteration_count += 1;
    }

    if iteration_count % 2 == 0 {
        tier1::copy(device, &flip, Some(dst.clone()))?;
    } else {
        tier1::copy(device, &flop, Some(dst.clone()))?;
    }
    Ok(dst)
}
