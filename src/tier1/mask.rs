use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::types::DType;

/// Apply a mask: set pixels to 0 where mask == 0.
pub fn mask(
    device: &DeviceArc,
    src: &ArrayPtr,
    mask_arr: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let kernel = ("mask", include_str!("../../kernels/mask.cl"));
    let params = vec![
        ("src0", ParameterValue::Array(src.clone())),
        ("src1", ParameterValue::Array(mask_arr.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    execute(device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}
