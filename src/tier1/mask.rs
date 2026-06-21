use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;

/// Apply a mask: set pixels to 0 where mask == 0.
pub fn mask(
    device: &DeviceArc,
    src: &ArrayPtr,
    mask_arr: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like_same(src, dst, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("mask", ParameterValue::Array(mask_arr.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        ("mask", include_str!("../../kernels/mask.cl")),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
