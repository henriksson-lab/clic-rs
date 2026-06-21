use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier4;
use crate::tier5;
use crate::types::LABEL;

/// Label objects from a gray-value image using Gaussian blur, Otsu thresholding,
/// and connected-component labeling.
///
/// Mirrors CLIc's `gauss_otsu_labeling_func`.
pub fn gauss_otsu_labeling(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    outline_sigma: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    let temp = tier1::gaussian_blur(
        device,
        src,
        None,
        outline_sigma,
        outline_sigma,
        outline_sigma,
    )?;
    let binary = tier4::threshold_otsu(device, &temp, None)?;
    tier5::connected_component_labeling(device, &binary, Some(dst), "box")
}
