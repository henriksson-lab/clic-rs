use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::utils::shape_to_dimension;

/// Sum contiguous blocks along the active dimension selected from `src` shape.
///
/// Mirrors CLIc's `sum_reduction_x`: when `dst` is omitted, the destination
/// size is reduced by `blocksize` along x for 1D input, y for 2D input, and z
/// for 3D input. The upstream implementation notes that only 1D data was
/// tested.
pub fn sum_reduction_x(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    blocksize: i32,
) -> Result<ArrayPtr> {
    let dst = if let Some(dst) = dst {
        dst
    } else {
        let (mut dst_width, mut dst_height, mut dst_depth, dtype, mtype, src_device) = {
            let src = src.lock().unwrap();
            (
                src.width(),
                src.height(),
                src.depth(),
                src.dtype(),
                src.mtype(),
                src.device().clone(),
            )
        };
        let dim = shape_to_dimension(dst_width, dst_height, dst_depth);
        let blocksize = blocksize as usize;
        match dim {
            1 => dst_width /= blocksize,
            2 => dst_height /= blocksize,
            3 => dst_depth /= blocksize,
            _ => {}
        }
        Array::create(
            dst_width,
            dst_height,
            dst_depth,
            1,
            dtype,
            mtype,
            &src_device,
        )?
    };
    let kernel = (
        "sum_reduction_x",
        include_str!("../../kernels/sum_reduction_x.cl"),
    );
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("index", ParameterValue::Int(blocksize)),
    ];
    let range = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
