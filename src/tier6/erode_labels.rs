use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::types::LABEL;
use crate::{tier1, tier4, tier5};

/// Erodes labels to a smaller size.
///
/// Depending on the label image and radius, labels may disappear or split into
/// multiple islands. Thus, overlapping labels of input and output may not have
/// the same identifier. This operation assumes input images are isotropic.
pub fn erode_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
    relabel: bool,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    if radius <= 0 {
        return tier1::copy(device, src, Some(dst));
    }

    let mut temp = tier1::detect_label_edges(device, src, None)?;
    let temp1 = tier1::binary_not(device, &temp, None)?;
    temp = tier1::mask(device, src, &temp1, Some(temp))?;
    drop(temp1);

    if radius == 1 {
        tier1::copy(device, &temp, Some(dst.clone()))?;
    }

    for i in 0..(radius - 1) {
        let (active, passive) = if i % 2 == 0 {
            (&temp, &dst)
        } else {
            (&dst, &temp)
        };
        let connectivity = if i % 2 == 0 { "sphere" } else { "box" };
        tier1::minimum_filter(
            device,
            active,
            Some(passive.clone()),
            1.0,
            1.0,
            1.0,
            connectivity,
        )?;
    }

    if relabel {
        if radius % 2 != 0 {
            tier1::copy(device, &temp, Some(dst.clone()))?;
        }
        temp = tier1::not_equal_constant(device, &dst, Some(temp), 0.0)?;
        tier5::connected_component_labeling(device, &temp, Some(dst.clone()), "sphere")?;
    } else {
        if radius % 2 == 0 {
            tier1::copy(device, &dst, Some(temp.clone()))?;
        }
        tier4::relabel_sequential(device, &temp, Some(dst.clone()), 4096)?;
    }
    Ok(dst)
}
