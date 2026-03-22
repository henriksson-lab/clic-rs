//! Tier 5 — higher-level operations composing lower tiers.

use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier4;

/// Return true if both arrays have equal shapes and all pixel values are identical.
pub fn array_equal(device: &DeviceArc, src0: &ArrayPtr, src1: &ArrayPtr) -> Result<bool> {
    let (w0, h0, d0) = { let l = src0.lock().unwrap(); (l.width(), l.height(), l.depth()) };
    let (w1, h1, d1) = { let l = src1.lock().unwrap(); (l.width(), l.height(), l.depth()) };
    if w0 != w1 || h0 != h1 || d0 != d1 {
        return Ok(false);
    }
    let mse = tier4::mean_squared_error(device, src0, src1)?;
    Ok(mse == 0.0)
}
