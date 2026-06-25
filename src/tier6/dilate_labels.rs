use crate::array::{Array, ArrayPtr};
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
    let flop = Array::create_from_array(&flip)?;
    let flag = Array::create(1, 1, 1, 1, DType::Float, MType::Buffer, device)?;
    flag.lock().unwrap().fill(0.0)?;

    let mut iter_count = 0;
    let mut flag_value = 1.0_f32;
    while flag_value > 0.0 && iter_count < radius {
        let active = if iter_count % 2 == 0 { &flip } else { &flop };
        let passive = if iter_count % 2 == 0 { &flop } else { &flip };
        tier1::onlyzero_overwrite_maximum(
            device,
            active,
            &flag,
            Some(passive.clone()),
            if iter_count % 2 == 0 { "box" } else { "sphere" },
        )?;

        flag.lock()
            .unwrap()
            .read_to(std::slice::from_mut(&mut flag_value))?;
        if flag_value > 0.0 {
            flag.lock().unwrap().fill(0.0)?;
        }
        iter_count += 1;
    }

    tier1::copy(
        device,
        if iter_count % 2 == 0 { &flip } else { &flop },
        Some(dst),
    )
}
