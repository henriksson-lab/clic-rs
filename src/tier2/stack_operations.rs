use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;

/// Crop a volume into a new volume along the z axis.
///
/// Mirrors CLIc's `sub_stack_func`.
pub fn sub_stack(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    start_z: i32,
    end_z: i32,
) -> Result<ArrayPtr> {
    let nb_slice = (end_z - start_z + 1).max(1) as usize;

    tier1::crop(device, src, dst, 0, 0, start_z, 0, 0, nb_slice)
}

/// Reduce the number of z slices by keeping every `reduction_factor`-th slice.
///
/// Mirrors CLIc's `reduce_stack_func`.
pub fn reduce_stack(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    reduction_factor: i32,
    offset: i32,
) -> Result<ArrayPtr> {
    let reduction_factor = reduction_factor.max(1);
    let (width, height, depth, dtype) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.depth(), src.dtype)
    };
    let num_slice = depth / reduction_factor as usize;

    let dst = tier0::create_dst(src, dst, width, height, num_slice, dtype, device)?;
    let temp_slice = {
        let src = src.lock().unwrap();
        Array::create(width, height, 1, 2, src.dtype, src.mtype, device)?
    };

    for z in 0..num_slice {
        let src_z = z as i32 * reduction_factor + offset;
        tier1::copy_slice(device, src, Some(temp_slice.clone()), src_z)?;
        tier1::copy_slice(device, &temp_slice, Some(dst.clone()), z as i32)?;
    }

    Ok(dst)
}
