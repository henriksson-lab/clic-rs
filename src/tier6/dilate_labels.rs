use crate::array::{pull, Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::types::{DType, MType, LABEL};

/// Dilates labels to a larger size without overwriting neighboring labels.
///
/// Similar to the implementation in scikit-image and MorphoLibJ. This
/// operation assumes input images are isotropic.
pub fn dilate_labels(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    radius: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    if radius <= 0 {
        return tier1::copy(device, src, Some(dst));
    }

    let flip = tier1::copy(device, src, None)?;
    let flop = Array::create_like(&flip, device)?;
    let flag = Array::create(1, 1, 1, 1, DType::Int32, MType::Buffer, device)?;
    flag.lock().unwrap().fill(0.0)?;

    let mut iter_count = 0;
    let mut flag_value = 1_i32;
    while flag_value > 0 && iter_count < radius {
        let (active, passive, connectivity) = if iter_count % 2 == 0 {
            (&flip, &flop, "box")
        } else {
            (&flop, &flip, "sphere")
        };
        tier1::onlyzero_overwrite_maximum(
            device,
            active,
            &flag,
            Some(passive.clone()),
            connectivity,
        )?;

        let flag_host: Vec<i32> = pull(&flag)?;
        flag_value = flag_host[0];
        if flag_value > 0 {
            flag.lock().unwrap().fill(0.0)?;
        }
        iter_count += 1;
    }

    let src = if iter_count % 2 == 0 { &flip } else { &flop };
    tier1::copy(device, src, Some(dst))
}
