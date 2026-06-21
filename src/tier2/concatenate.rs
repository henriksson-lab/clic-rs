use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier0;
use crate::tier1;
use crate::types::DType;

/// Concatenate two arrays along an axis (0: x, 1: y, 2: z).
///
/// Mirrors CLIc's `concatenate_func`: allocate the destination from `src0`,
/// clear it, then paste `src0` and `src1` into their axis-specific positions.
pub fn concatenate(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
    axis: i32,
) -> Result<ArrayPtr> {
    let (src0_width, src0_height, src0_depth) = {
        let s0 = src0.lock().unwrap();
        (s0.width(), s0.height(), s0.depth())
    };
    let (src1_width, src1_height, src1_depth) = {
        let s1 = src1.lock().unwrap();
        (s1.width(), s1.height(), s1.depth())
    };

    let (dst_width, dst_height, dst_depth, src1_x, src1_y, src1_z) = match axis {
        0 => (
            src0_width + src1_width,
            src0_height,
            src0_depth,
            src0_width as i32,
            0,
            0,
        ),
        1 => (
            src0_width,
            src0_height + src1_height,
            src0_depth,
            0,
            src0_height as i32,
            0,
        ),
        2 => (
            src0_width,
            src0_height,
            src0_depth + src1_depth,
            0,
            0,
            src0_depth as i32,
        ),
        _ => {
            return Err(CleError::Other(
                "concatenate: axis must be 0, 1 or 2".into(),
            ));
        }
    };

    let dst = tier0::create_dst(
        src0,
        dst,
        dst_width,
        dst_height,
        dst_depth,
        DType::Unknown,
        device,
    )?;
    dst.lock().unwrap().fill(0.0)?;

    tier1::paste(device, src0, Some(dst.clone()), 0, 0, 0)?;
    tier1::paste(device, src1, Some(dst.clone()), src1_x, src1_y, src1_z)?;
    Ok(dst)
}

/// Concatenate two images or stacks along the x axis.
pub fn concatenate_along_x(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    concatenate(device, src0, src1, dst, 0)
}

/// Concatenate two images or stacks along the y axis.
pub fn concatenate_along_y(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    concatenate(device, src0, src1, dst, 1)
}

/// Concatenate two images or stacks along the z axis.
pub fn concatenate_along_z(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    concatenate(device, src0, src1, dst, 2)
}
