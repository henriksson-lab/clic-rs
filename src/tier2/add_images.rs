use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

/// src0 + src1 element-wise.
pub fn add_images(
    device: &DeviceArc,
    src0: &ArrayPtr,
    src1: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    tier1::add_images_weighted(device, src0, src1, dst, 1.0, 1.0)
}
