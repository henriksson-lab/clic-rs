use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::types::{DType, MType, LABEL};
use crate::{tier1, tier5};

/// Takes a binary image, labels connected components, and dilates the regions
/// using an octagon shape until they touch. The region growing is limited to a
/// masked area. The resulting label map is written to the output.
///
/// Mirrors CLIc's `masked_voronoi_labeling_func`.
///
/// See: https://clij.github.io/clij2-docs/reference_maskedVoronoiLabeling
pub fn masked_voronoi_labeling(
    device: &DeviceArc,
    src: &ArrayPtr,
    mask: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    let flip = tier0::create_like(src, None, DType::Float, device)?;
    let flop = tier0::create_like(src, None, DType::Float, device)?;
    let flup = tier0::create_like(src, None, DType::Float, device)?;

    tier1::add_image_and_scalar(device, mask, Some(flup.clone()), -1.0)?;
    tier5::connected_component_labeling(device, src, Some(flop.clone()), "box")?;
    tier1::add_images_weighted(device, &flop, &flup, Some(flip.clone()), 1.0, 1.0)?;

    let flag = Array::create(1, 1, 1, 1, DType::Int32, MType::Buffer, device)?;
    flag.lock().unwrap().fill(1.0)?;

    let mut flag_value = 1_i32;
    let mut iter_count = 0;
    while flag_value > 0 {
        let active = if iter_count % 2 == 0 { &flip } else { &flop };
        let passive = if iter_count % 2 == 0 { &flop } else { &flip };
        let connectivity = if iter_count % 2 == 0 { "box" } else { "sphere" };
        tier1::onlyzero_overwrite_maximum(
            device,
            active,
            &flag,
            Some(passive.clone()),
            connectivity,
        )?;

        flag.lock()
            .unwrap()
            .read_to(std::slice::from_mut(&mut flag_value))?;
        if flag_value > 0 {
            flag.lock().unwrap().fill(0.0)?;
        }
        iter_count += 1;
    }

    tier1::mask(
        device,
        if iter_count % 2 == 0 { &flip } else { &flop },
        mask,
        Some(dst),
    )
}
