use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::{DType, INDEX};

const STD_PROJECTION_SRC: &str = r#"
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void std_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    int axis  // 0=X projection, 1=Y projection, 2=Z projection
)
{
  const int id0 = get_global_id(0);
  const int id1 = get_global_id(1);

  // Projection length along the target axis
  const int n = (axis == 0) ? GET_IMAGE_WIDTH(src) :
                (axis == 1) ? GET_IMAGE_HEIGHT(src) : GET_IMAGE_DEPTH(src);

  // Welford's online algorithm: single pass, numerically stable
  float mean = 0;
  float m2 = 0;

  for (int i = 0; i < n; i++)
  {
    // Map (id0, id1, i) to (x, y, z) based on projection axis
    const int x = (axis == 0) ? i   : id0;
    const int y = (axis == 0) ? id0 : (axis == 1) ? i : id1;
    const int z = (axis == 2) ? i   : id1;

    const float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    const float delta = value - mean;
    mean += delta / (float)(i + 1);
    const float delta2 = value - mean;
    m2 += delta * delta2;
  }

  const float std_value = (n > 1) ? sqrt(m2 / (float)(n - 1)) : 0;

  // Output coordinates based on create_* output layout:
  // X projection: create_zy -> (depth, height, 1) range {height, width, 1}
  // Y projection: create_xz -> (width, depth, 1) range {width, depth, 1}
  // Z projection: create_xy -> (width, height, 1) range {width, height, 1}
  const int ox = (axis == 0) ? id1 : id0;
  const int oy = (axis == 0) ? id0 : id1;
  const int oz = 0;

  WRITE_IMAGE(dst, POS_dst_INSTANCE(ox, oy, oz, 0), CONVERT_dst_PIXEL_TYPE(std_value));
}
"#;

pub fn std_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_zy(src, dst, DType::Float, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("axis", ParameterValue::Int(0)),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), 1]
    };
    execute(
        device,
        ("std_projection", STD_PROJECTION_SRC),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

pub fn std_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xz(src, dst, DType::Float, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("axis", ParameterValue::Int(1)),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), 1]
    };
    execute(
        device,
        ("std_projection", STD_PROJECTION_SRC),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

pub fn std_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xy(src, dst, DType::Float, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("axis", ParameterValue::Int(2)),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), 1]
    };
    execute(
        device,
        ("std_projection", STD_PROJECTION_SRC),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

pub fn maximum_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_zy(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(0))];
    execute(
        device,
        (
            "maximum_projection",
            include_str!("../../kernels/maximum_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn maximum_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xz(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(1))];
    execute(
        device,
        (
            "maximum_projection",
            include_str!("../../kernels/maximum_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn maximum_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xy(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(2))];
    execute(
        device,
        (
            "maximum_projection",
            include_str!("../../kernels/maximum_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn mean_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_zy(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(0))];
    execute(
        device,
        (
            "mean_projection",
            include_str!("../../kernels/mean_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn mean_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xz(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(1))];
    execute(
        device,
        (
            "mean_projection",
            include_str!("../../kernels/mean_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn mean_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xy(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(2))];
    execute(
        device,
        (
            "mean_projection",
            include_str!("../../kernels/mean_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn minimum_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_zy(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(0))];
    execute(
        device,
        (
            "minimum_projection",
            include_str!("../../kernels/minimum_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn minimum_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xz(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(1))];
    execute(
        device,
        (
            "minimum_projection",
            include_str!("../../kernels/minimum_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn minimum_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xy(src, dst, DType::Unknown, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(2))];
    execute(
        device,
        (
            "minimum_projection",
            include_str!("../../kernels/minimum_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn sum_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_zy(src, dst, DType::Float, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(0))];
    execute(
        device,
        (
            "sum_projection",
            include_str!("../../kernels/sum_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn sum_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xz(src, dst, DType::Float, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(1))];
    execute(
        device,
        (
            "sum_projection",
            include_str!("../../kernels/sum_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn sum_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xy(src, dst, DType::Float, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(2))];
    execute(
        device,
        (
            "sum_projection",
            include_str!("../../kernels/sum_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &constants,
    )?;
    Ok(dst)
}

pub fn x_position_of_maximum_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_zy(src, dst, INDEX, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(
        device,
        (
            "x_position_of_maximum_x_projection",
            include_str!("../../kernels/x_position_of_maximum_x_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

pub fn x_position_of_minimum_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_zy(src, dst, INDEX, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(
        device,
        (
            "x_position_of_minimum_x_projection",
            include_str!("../../kernels/x_position_of_minimum_x_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

pub fn y_position_of_maximum_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xz(src, dst, INDEX, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(
        device,
        (
            "y_position_of_maximum_y_projection",
            include_str!("../../kernels/y_position_of_maximum_y_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

pub fn y_position_of_minimum_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xz(src, dst, INDEX, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(
        device,
        (
            "y_position_of_minimum_y_projection",
            include_str!("../../kernels/y_position_of_minimum_y_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

pub fn z_position_of_maximum_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xy(src, dst, INDEX, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(
        device,
        (
            "z_position_of_maximum_z_projection",
            include_str!("../../kernels/z_position_of_maximum_z_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

pub fn z_position_of_minimum_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xy(src, dst, INDEX, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    execute(
        device,
        (
            "z_position_of_minimum_z_projection",
            include_str!("../../kernels/z_position_of_minimum_z_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

pub fn z_position_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    position: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xy(src, dst, DType::Unknown, device)?;
    let range = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("position", ParameterValue::Array(position.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        (
            "z_position_projection",
            include_str!("../../kernels/z_position_projection.cl"),
        ),
        &params,
        range,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
