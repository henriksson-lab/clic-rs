use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier2;
use crate::types::LABEL;

/// Create a vector marking which label values are present in `src`.
///
/// Mirrors CLIc's `flag_existing_labels_func`.
pub fn flag_existing_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let max_label = tier2::maximum_of_all_pixels(device, src)? as usize;
    let dst = tier0::create_vector_like(src, dst, max_label + 1, LABEL, device)?;
    dst.lock().unwrap().fill(0.0)?;

    let global = {
        let lock = src.lock().unwrap();
        [lock.width(), lock.height(), lock.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        (
            "flag_existing_labels",
            include_str!("../../kernels/flag_existing_labels.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;

    Ok(dst)
}
