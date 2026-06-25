use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Reduce each x/y column to the minimum source value where the mask is non-zero.
///
/// The reduced source and mask outputs are XY images; the mask output is set to
/// one where at least one masked pixel contributed and zero otherwise. Mirrors
/// CLIc's `minimum_of_masked_pixels_reduction_func`.
pub fn minimum_of_masked_pixels_reduction(
    device: &DeviceArc,
    src: &ArrayPtr,
    mask: &ArrayPtr,
    reduced_src: Option<ArrayPtr>,
    reduced_mask: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let reduced_src = tier0::create_xy(src, reduced_src, DType::Unknown, device)?;
    let reduced_mask = tier0::create_xy(mask, reduced_mask, DType::Unknown, device)?;
    let kernel = (
        "minimum_of_masked_pixels_reduction",
        include_str!("../../kernels/minimum_of_masked_pixels_reduction.cl"),
    );
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("mask", ParameterValue::Array(mask.clone())),
        ("dst_src", ParameterValue::Array(reduced_src.clone())),
        ("dst_mask", ParameterValue::Array(reduced_mask)),
    ];
    let range = {
        let l = reduced_src.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(reduced_src)
}
