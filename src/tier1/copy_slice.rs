use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

fn shape(arr: &ArrayPtr) -> [usize; 3] {
    let l = arr.lock().unwrap();
    [l.width(), l.height(), l.depth()]
}

fn copy_slice_axis(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    slice_index: i32,
    to_kernel: (&'static str, &'static str),
    from_kernel: (&'static str, &'static str),
    to_global: impl Fn(&ArrayPtr, &ArrayPtr) -> [usize; 3],
    from_global: impl Fn(&ArrayPtr, &ArrayPtr) -> [usize; 3],
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index", ParameterValue::Int(slice_index)),
    ];
    let dst_depth = {
        let d = dst.lock().unwrap();
        d.depth()
    };
    let (kernel, global) = if dst_depth > 1 {
        (to_kernel, to_global(src, &dst))
    } else {
        (from_kernel, from_global(src, &dst))
    };
    execute(device, kernel, &params, global, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Copy a 2D image into a z slice of a 3D stack, or copy a z slice into a 2D image.
///
/// Mirrors CLIc's `copy_slice_func`.
pub fn copy_slice(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    slice_index: i32,
) -> Result<ArrayPtr> {
    copy_slice_axis(
        device,
        src,
        dst,
        slice_index,
        (
            "copy_slice_to",
            include_str!("../../kernels/copy_slice_to.cl"),
        ),
        (
            "copy_slice_from",
            include_str!("../../kernels/copy_slice_from.cl"),
        ),
        |src, _| {
            let [width, height, _] = shape(src);
            [width, height, 1]
        },
        |_, dst| shape(dst),
    )
}

/// Copy a 2D image into a y slice of a 3D stack, or copy a y slice into a 2D image.
///
/// Mirrors CLIc's `copy_horizontal_slice_func`.
pub fn copy_horizontal_slice(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    slice_index: i32,
) -> Result<ArrayPtr> {
    copy_slice_axis(
        device,
        src,
        dst,
        slice_index,
        (
            "copy_horizontal_slice_to",
            include_str!("../../kernels/copy_horizontal_slice_to.cl"),
        ),
        (
            "copy_horizontal_slice_from",
            include_str!("../../kernels/copy_horizontal_slice_from.cl"),
        ),
        |_, dst| shape(dst),
        |_, dst| shape(dst),
    )
}

/// Copy a 2D image into an x slice of a 3D stack, or copy an x slice into a 2D image.
///
/// Mirrors CLIc's `copy_vertical_slice_func`.
pub fn copy_vertical_slice(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    slice_index: i32,
) -> Result<ArrayPtr> {
    copy_slice_axis(
        device,
        src,
        dst,
        slice_index,
        (
            "copy_vertical_slice_to",
            include_str!("../../kernels/copy_vertical_slice_to.cl"),
        ),
        (
            "copy_vertical_slice_from",
            include_str!("../../kernels/copy_vertical_slice_from.cl"),
        ),
        |src, _| {
            let [width, height, _] = shape(src);
            [width, height, 1]
        },
        |_, dst| shape(dst),
    )
}
