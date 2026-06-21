use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier6;
use crate::types::LABEL;

/// Apply morphological closing to a label image.
///
/// Mirrors CLIc's `closing_labels_func`.
pub fn closing_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    if radius == 0 {
        return tier1::copy(device, src, Some(dst));
    }

    let temp = tier6::dilate_labels(device, src, None, radius)?;
    let flip = tier1::greater_constant(device, &temp, None, 0.0)?;
    let flop = tier0::create_like_same(&flip, None, device)?;

    for i in 0..radius {
        let (active, passive, connectivity) = if i % 2 == 0 {
            (&flip, &flop, "sphere")
        } else {
            (&flop, &flip, "box")
        };
        tier1::binary_erode(
            device,
            active,
            Some(passive.clone()),
            1.0,
            1.0,
            1.0,
            connectivity,
        )?;
    }

    let mask = if radius % 2 == 0 { &flip } else { &flop };
    tier1::multiply_images(device, mask, &temp, Some(dst))
}
