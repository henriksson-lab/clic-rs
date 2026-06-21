use crate::array::{pull, Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ParameterValue};
use crate::tier0;
use crate::tier1;
use crate::types::{DType, MType, LABEL};

const POINTLIST_WITH_LABELS_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void pointlist_with_labels(
    IMAGE_src_TYPE src,
    IMAGE_dst_TYPE dst
)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float value = (float)(x + 1);
  if (y < GET_IMAGE_HEIGHT(src)) {
    value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,0,0)).x;
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(x,y,0,0), CONVERT_dst_PIXEL_TYPE(value));
}
"#;

/// Transform a binary spots image into a same-sized label image.
///
/// Mirrors CLIc's `label_spots_func`.
pub fn label_spots(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    dst.lock().unwrap().fill(0.0)?;

    let spot_count_in_x = tier1::sum_x_projection(device, src, None)?;
    let spot_count_in_xy = tier1::sum_y_projection(device, &spot_count_in_x, None)?;
    let global = {
        let dst = dst.lock().unwrap();
        [1, dst.height(), dst.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("countX", ParameterValue::Array(spot_count_in_x)),
        ("countXY", ParameterValue::Array(spot_count_in_xy)),
    ];
    execute(
        device,
        (
            "label_spots_in_x",
            include_str!("../../kernels/label_spots_in_x.cl"),
        ),
        &params,
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

fn create_labelled_spots_dst(device: &DeviceArc, src: &ArrayPtr) -> Result<ArrayPtr> {
    let ndims = {
        let src = src.lock().unwrap();
        src.height()
    };
    let max = tier1::maximum_x_projection(device, src, None)?;
    let max_value: Vec<f32> = pull(&max)?;

    let width = if ndims > 0 {
        max_value[0] as usize + 1
    } else {
        1
    };
    let height = if ndims > 1 {
        max_value[1] as usize + 1
    } else {
        1
    };
    let depth = if ndims > 2 {
        max_value[2] as usize + 1
    } else {
        1
    };

    Array::create(width, height, depth, ndims, LABEL, MType::Buffer, device)
}

fn pointlist_with_labels(device: &DeviceArc, src: &ArrayPtr) -> Result<ArrayPtr> {
    let (width, height) = {
        let src = src.lock().unwrap();
        (src.width(), src.height())
    };
    let dst = Array::create(width, height + 1, 1, 2, DType::Float, MType::Buffer, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        ("pointlist_with_labels", POINTLIST_WITH_LABELS_SRC),
        &params,
        [width, height + 1, 1],
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

/// Convert a pointlist image into labelled spots in a label image.
///
/// Mirrors CLIc's `pointlist_to_labelled_spots_func`.
pub fn pointlist_to_labelled_spots(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = match dst {
        Some(dst) => dst,
        None => create_labelled_spots_dst(device, src)?,
    };
    dst.lock().unwrap().fill(0.0)?;

    let labelled_pointlist = pointlist_with_labels(device, src)?;
    tier1::write_values_to_positions(device, &labelled_pointlist, Some(dst))
}
