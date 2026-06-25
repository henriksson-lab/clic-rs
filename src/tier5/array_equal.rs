use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier4;

/// Return true if both arrays have equal shapes and all pixel values are identical.
pub fn array_equal(device: &DeviceArc, src0: &ArrayPtr, src1: &ArrayPtr) -> Result<bool> {
    let size0 = src0.lock().unwrap().size();
    let size1 = src1.lock().unwrap().size();
    if size0 != size1 {
        return Ok(false);
    }

    let (w0, h0, d0) = {
        let src0 = src0.lock().unwrap();
        (src0.width(), src0.height(), src0.depth())
    };
    let (w1, h1, d1) = {
        let src1 = src1.lock().unwrap();
        (src1.width(), src1.height(), src1.depth())
    };
    if w0 != w1 || h0 != h1 || d0 != d1 {
        return Ok(false);
    }

    if size0 == 0 && size1 == 0 {
        return Ok(true);
    }
    let mse = tier4::mean_squared_error(device, src0, src1)?;
    Ok(mse == 0.0)
}
