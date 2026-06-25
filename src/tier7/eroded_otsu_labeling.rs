use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::types::{DType, LABEL};
use crate::{tier1, tier4, tier6};

/// Segment and label an image using blurring, Otsu thresholding, binary erosion,
/// and masked Voronoi labeling.
///
/// After blurring and Otsu thresholding the image, iterative binary erosion is
/// applied. Objects in the eroded image are labeled, and the labels are extended
/// back into the initial binary image using masked Voronoi labeling.
///
/// This function is similar to `voronoi_otsu_labeling`. It is intended to deal
/// better with dense objects where labels can swap into each other. As with
/// Voronoi-Otsu labeling, small objects may disappear.
///
/// Mirrors CLIc's `eroded_otsu_labeling_func`.
///
/// See: https://github.com/biovoxxel/bv3dbox
/// See: https://zenodo.org/badge/latestdoi/434949702
pub fn eroded_otsu_labeling(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    number_of_erosions: i32,
    outline_sigma: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    let blurred = tier1::gaussian_blur(
        device,
        src,
        None,
        outline_sigma,
        outline_sigma,
        outline_sigma,
    )?;
    let binary = tier4::threshold_otsu(device, &blurred, None)?;

    let mut eroded1 = tier0::create_like(&binary, None, DType::Unknown, device)?;
    let mut eroded2 = tier0::create_like(&binary, None, DType::Unknown, device)?;
    binary.lock().unwrap().copy_to(&eroded1)?;

    for _i in 0..number_of_erosions {
        tier1::binary_erode(
            device,
            &eroded1,
            Some(eroded2.clone()),
            1.0,
            1.0,
            1.0,
            "box",
        )?;
        std::mem::swap(&mut eroded1, &mut eroded2);
    }

    tier6::masked_voronoi_labeling(device, &eroded1, &binary, Some(dst))
}
