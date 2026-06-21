use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier2;
use crate::types::LABEL;
use crate::utils::shape_to_dimension;

/// Convert a labelled spots image to a point-list array.
///
/// Mirrors CLIc's `labelled_spots_to_pointlist_func`.
pub fn labelled_spots_to_pointlist(
    device: &DeviceArc,
    label: &ArrayPtr,
    pointlist: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let max_label = tier2::maximum_of_all_pixels(device, label)? as usize;
    let (width, height, depth) = {
        let lock = label.lock().unwrap();
        (lock.width(), lock.height(), lock.depth())
    };
    let dim = shape_to_dimension(width, height, depth);
    let pointlist = tier0::create_dst(label, pointlist, max_label, dim, 1, LABEL, device)?;
    pointlist.lock().unwrap().fill(0.0)?;

    let params = vec![
        ("src", ParameterValue::Array(label.clone())),
        ("dst", ParameterValue::Array(pointlist.clone())),
    ];
    execute(
        device,
        (
            "labelled_spots_to_point_list",
            include_str!("../../kernels/labelled_spots_to_point_list.cl"),
        ),
        &params,
        [width, height, depth],
        [0, 0, 0],
        &[],
    )?;

    Ok(pointlist)
}
