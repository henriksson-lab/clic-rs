use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier2;
use crate::tier5;
use crate::types::LABEL;

/// Label connected components and extend labels by Voronoi growing.
///
/// Mirrors CLIc's `voronoi_labeling_func`.
pub fn voronoi_labeling(
    device: &DeviceArc,
    input_binary: &ArrayPtr,
    output_labels: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let output_labels = tier0::create_like(input_binary, output_labels, LABEL, device)?;
    let flip = tier5::connected_component_labeling(device, input_binary, None, "box")?;
    tier2::extend_labeling_via_voronoi(device, &flip, Some(output_labels))
}
