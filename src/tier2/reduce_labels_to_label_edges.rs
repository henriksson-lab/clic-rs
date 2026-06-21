use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

pub fn reduce_labels_to_label_edges(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let binary = tier1::detect_label_edges(device, src, None)?;
    tier1::mask(device, src, &binary, dst)
}
