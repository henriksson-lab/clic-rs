use crate::array::Array;
use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::{tier1, tier2, tier3};

pub fn relabel_sequential(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    blocksize: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let max_label = tier2::maximum_of_all_pixels(device, src)? as usize;
    let (dtype, mtype) = {
        let src = src.lock().unwrap();
        (src.dtype(), src.mtype())
    };
    let flagged = Array::create(max_label + 1, 1, 1, 1, dtype, mtype, device)?;
    flagged.lock().unwrap().fill(0.0)?;
    tier3::flag_existing_labels(device, src, Some(flagged.clone()))?;
    tier1::set_column(device, &flagged, 0, 0.0)?;

    let blocksize_usize = blocksize.max(1) as usize;
    let block_sums = Array::create(
        ((max_label + 1) / blocksize_usize) + 1,
        1,
        1,
        1,
        dtype,
        mtype,
        device,
    )?;
    tier1::sum_reduction_x(device, &flagged, Some(block_sums.clone()), blocksize)?;
    let new_indices = Array::create(max_label + 1, 1, 1, 1, dtype, mtype, device)?;
    tier1::block_enumerate(
        device,
        &flagged,
        &block_sums,
        Some(new_indices.clone()),
        blocksize,
    )?;
    tier1::replace_values(device, src, &new_indices, Some(dst.clone()))?;
    Ok(dst)
}
