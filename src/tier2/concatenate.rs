use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::tier0;
use crate::tier1;

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
    let mut dst = dst;
    match axis {
        0 => {
            let (src0_width, src0_height, src0_depth, src0_dtype) = {
                let src0 = src0.lock().unwrap();
                (src0.width(), src0.height(), src0.depth(), src0.dtype)
            };
            let src1_width = src1.lock().unwrap().width();
            dst = Some(tier0::create_dst(
                src0,
                dst,
                src0_width + src1_width,
                src0_height,
                src0_depth,
                src0_dtype,
                device,
            )?);
            let dst_ref = dst.as_ref().unwrap();
            dst_ref.lock().unwrap().fill(0.0)?;
            tier1::paste(device, src0, Some(dst_ref.clone()), 0, 0, 0)?;
            tier1::paste(device, src1, Some(dst_ref.clone()), src0_width as i32, 0, 0)?;
        }
        1 => {
            let (src0_width, src0_height, src0_depth, src0_dtype) = {
                let src0 = src0.lock().unwrap();
                (src0.width(), src0.height(), src0.depth(), src0.dtype)
            };
            let src1_height = src1.lock().unwrap().height();
            dst = Some(tier0::create_dst(
                src0,
                dst,
                src0_width,
                src0_height + src1_height,
                src0_depth,
                src0_dtype,
                device,
            )?);
            let dst_ref = dst.as_ref().unwrap();
            dst_ref.lock().unwrap().fill(0.0)?;
            tier1::paste(device, src0, Some(dst_ref.clone()), 0, 0, 0)?;
            tier1::paste(
                device,
                src1,
                Some(dst_ref.clone()),
                0,
                src0_height as i32,
                0,
            )?;
        }
        2 => {
            let (src0_width, src0_height, src0_depth, src0_dtype) = {
                let src0 = src0.lock().unwrap();
                (src0.width(), src0.height(), src0.depth(), src0.dtype)
            };
            let src1_depth = src1.lock().unwrap().depth();
            dst = Some(tier0::create_dst(
                src0,
                dst,
                src0_width,
                src0_height,
                src0_depth + src1_depth,
                src0_dtype,
                device,
            )?);
            let dst_ref = dst.as_ref().unwrap();
            dst_ref.lock().unwrap().fill(0.0)?;
            tier1::paste(device, src0, Some(dst_ref.clone()), 0, 0, 0)?;
            tier1::paste(device, src1, Some(dst_ref.clone()), 0, 0, src0_depth as i32)?;
        }
        _ => {
            return Err(CleError::Other(
                "concatenate: axis must be 0, 1 or 2".into(),
            ));
        }
    };
    Ok(dst.unwrap())
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
