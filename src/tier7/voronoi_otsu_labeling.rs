use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::types::LABEL;
use crate::{tier1, tier2, tier4, tier6};

/// Segment and label an image using spot detection, Otsu thresholding, and
/// masked Voronoi labeling.
///
/// Spots are detected from a Gaussian-blurred image using local maxima. The
/// object outline is estimated by Gaussian blur followed by Otsu thresholding.
/// The spot seeds are restricted to the segmented area, expanded using masked
/// Voronoi labeling, and masked by the segmentation.
///
/// Mirrors CLIc's `voronoi_otsu_labeling_func`.
pub fn voronoi_otsu_labeling(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    spot_sigma: f32,
    outline_sigma: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;

    let spot_blurred = tier1::gaussian_blur(device, src, None, spot_sigma, spot_sigma, spot_sigma)?;
    let spot = tier2::detect_maxima(device, &spot_blurred, None, 0.0, 0.0, 0.0, "box")?;

    let outline_blurred = tier1::gaussian_blur(
        device,
        src,
        None,
        outline_sigma,
        outline_sigma,
        outline_sigma,
    )?;
    let segmentation = tier4::threshold_otsu(device, &outline_blurred, None)?;

    let binary = tier1::binary_and(device, &spot, &segmentation, None)?;
    let labeled = tier6::masked_voronoi_labeling(device, &binary, &segmentation, None)?;
    tier1::mask(device, &labeled, &segmentation, Some(dst))
}
