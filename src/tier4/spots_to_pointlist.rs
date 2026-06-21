use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::{tier2, tier3};

pub fn spots_to_pointlist(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let labeled_spots = tier2::label_spots(device, src, None)?;
    tier3::labelled_spots_to_pointlist(device, &labeled_spots, dst)
}
