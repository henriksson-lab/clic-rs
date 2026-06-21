use crate::array::{pull, Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::types::{DType, MType, LABEL};
use crate::{tier1, tier4};

/// Performs connected components analysis by inspecting the neighborhood of every pixel in a binary image and generates a label map.
///
/// `connectivity` defines the pixel neighborhood relationship and should be `"box"` or `"sphere"`.
/// Mirrors CLIc's `connected_component_labeling_func`.
pub fn connected_component_labeling(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    connectivity: &str,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    let temp1 = tier1::set_nonzero_pixels_to_pixelindex(device, src, None, 1)?;
    let temp2 = Array::create_like(&temp1, device)?;
    temp2.lock().unwrap().fill(0.0)?;

    let flag = Array::create(1, 1, 1, 1, DType::Int32, MType::Buffer, device)?;
    flag.lock().unwrap().fill(0.0)?;

    let mut flag_value = 1_i32;
    let mut iteration_count = 0;
    while flag_value > 0 {
        if iteration_count % 2 == 0 {
            tier1::nonzero_minimum(device, &temp1, &flag, Some(temp2.clone()), connectivity)?;
        } else {
            tier1::nonzero_minimum(device, &temp2, &flag, Some(temp1.clone()), connectivity)?;
        }

        let flag_host: Vec<i32> = pull(&flag)?;
        flag_value = flag_host[0];
        if flag_value > 0 {
            flag.lock().unwrap().fill(0.0)?;
        }
        iteration_count += 1;
    }

    let labeled = if iteration_count % 2 == 0 {
        temp1
    } else {
        temp2
    };
    tier4::relabel_sequential(device, &labeled, Some(dst), 4096)
}

/// Deprecated alias for [`connected_component_labeling`].
///
/// Mirrors CLIc's deprecated `connected_components_labeling_func`.
#[deprecated(note = "use connected_component_labeling instead")]
pub fn connected_components_labeling(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    connectivity: &str,
) -> Result<ArrayPtr> {
    connected_component_labeling(device, src, dst, connectivity)
}
