use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::types::DType;

/// Extend `src` to an explicit shape, filling uncovered pixels with `value`.
///
/// When `center` is true, the source is placed using the same ceil-half offset
/// as CLIc's `pad_func`; otherwise it is placed at the origin.
pub fn pad(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    value: f32,
    center: bool,
) -> Result<ArrayPtr> {
    let (src_width, src_height, src_depth) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.depth())
    };
    let dst = tier0::create_dst(src, dst, size_x, size_y, size_z, DType::Unknown, device)?;
    dst.lock().unwrap().fill(value)?;
    let (dst_width, dst_height, dst_depth) = {
        let dst = dst.lock().unwrap();
        (dst.width(), dst.height(), dst.depth())
    };

    let pad_x = src_width.abs_diff(size_x);
    let pad_y = src_height.abs_diff(size_y);
    let pad_z = src_depth.abs_diff(size_z);
    let mut offset = [0, 0, 0];
    if center {
        offset = [
            if dst_width > 1 { pad_x.div_ceil(2) } else { 0 },
            if dst_height > 1 { pad_y.div_ceil(2) } else { 0 },
            if dst_depth > 1 { pad_z.div_ceil(2) } else { 0 },
        ];
    }
    src.lock().unwrap().copy_to_region(
        &dst,
        [src_width, src_height, src_depth],
        [0, 0, 0],
        offset,
    )?;
    Ok(dst)
}

/// Extract an explicit shape from `src`, optionally centered like CLIc's `unpad_func`.
pub fn unpad(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    size_x: usize,
    size_y: usize,
    size_z: usize,
    center: bool,
) -> Result<ArrayPtr> {
    let (src_width, src_height, src_depth) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.depth())
    };
    let dst = tier0::create_dst(src, dst, size_x, size_y, size_z, DType::Unknown, device)?;
    let (dst_width, dst_height, dst_depth) = {
        let dst = dst.lock().unwrap();
        (dst.width(), dst.height(), dst.depth())
    };

    let pad_x = src_width.abs_diff(size_x);
    let pad_y = src_height.abs_diff(size_y);
    let pad_z = src_depth.abs_diff(size_z);
    let mut offset = [0, 0, 0];
    if center {
        offset = [
            if dst_width > 1 { pad_x.div_ceil(2) } else { 0 },
            if dst_height > 1 { pad_y.div_ceil(2) } else { 0 },
            if dst_depth > 1 { pad_z.div_ceil(2) } else { 0 },
        ];
    }
    src.lock().unwrap().copy_to_region(
        &dst,
        [dst_width, dst_height, dst_depth],
        offset,
        [0, 0, 0],
    )?;
    Ok(dst)
}
