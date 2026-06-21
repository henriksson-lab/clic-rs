use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::{DType, INDEX};

fn run_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: &ArrayPtr,
    axis: usize,
    kernel_name: &str,
    kernel_src: &str,
) -> Result<()> {
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![("PROJECTION_AXIS", ConstantValue::Int(axis as i32))];
    execute(
        device,
        (kernel_name, kernel_src),
        &params,
        global,
        [0, 0, 0],
        &constants,
    )
}

fn maximum_projection_axis(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    axis: usize,
) -> Result<ArrayPtr> {
    let dst = projection_dst(src, dst, axis, device)?;
    run_projection(
        device,
        src,
        &dst,
        axis,
        "maximum_projection",
        include_str!("../../kernels/maximum_projection.cl"),
    )?;
    Ok(dst)
}

fn minimum_projection_axis(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    axis: usize,
) -> Result<ArrayPtr> {
    let dst = projection_dst(src, dst, axis, device)?;
    run_projection(
        device,
        src,
        &dst,
        axis,
        "minimum_projection",
        include_str!("../../kernels/minimum_projection.cl"),
    )?;
    Ok(dst)
}

fn sum_projection_axis(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    axis: usize,
) -> Result<ArrayPtr> {
    let dst = projection_dst(src, dst, axis, device)?;
    run_projection(
        device,
        src,
        &dst,
        axis,
        "sum_projection",
        include_str!("../../kernels/sum_projection.cl"),
    )?;
    Ok(dst)
}

fn mean_projection_axis(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    axis: usize,
) -> Result<ArrayPtr> {
    let dst = projection_dst(src, dst, axis, device)?;
    run_projection(
        device,
        src,
        &dst,
        axis,
        "mean_projection",
        include_str!("../../kernels/mean_projection.cl"),
    )?;
    Ok(dst)
}

fn projection_dst(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    axis: usize,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    projection_dst_with_dtype(src, dst, axis, DType::Unknown, device)
}

fn projection_dst_with_dtype(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    axis: usize,
    dtype: DType,
    device: &DeviceArc,
) -> Result<ArrayPtr> {
    match axis {
        0 => tier0::create_zy(src, dst, dtype, device),
        1 => tier0::create_xz(src, dst, dtype, device),
        2 => tier0::create_xy(src, dst, dtype, device),
        _ => Err(crate::error::CleError::Other("Invalid axis".into())),
    }
}

fn position_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    axis: usize,
    kernel_name: &'static str,
    kernel_src: &'static str,
) -> Result<ArrayPtr> {
    let dst = projection_dst_with_dtype(src, dst, axis, INDEX, device)?;
    let global = {
        let l = dst.lock().unwrap();
        [l.width(), l.height(), l.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    execute(
        device,
        (kernel_name, kernel_src),
        &params,
        global,
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
    maximum_projection_axis(device, src, dst, 0)
}

pub fn maximum_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    maximum_projection_axis(device, src, dst, 1)
}

pub fn maximum_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    maximum_projection_axis(device, src, dst, 2)
}

pub fn minimum_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    minimum_projection_axis(device, src, dst, 0)
}

pub fn minimum_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    minimum_projection_axis(device, src, dst, 1)
}

pub fn minimum_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    minimum_projection_axis(device, src, dst, 2)
}

pub fn sum_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    sum_projection_axis(device, src, dst, 0)
}

pub fn sum_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    sum_projection_axis(device, src, dst, 1)
}

pub fn sum_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    sum_projection_axis(device, src, dst, 2)
}

pub fn mean_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    mean_projection_axis(device, src, dst, 0)
}

pub fn mean_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    mean_projection_axis(device, src, dst, 1)
}

pub fn mean_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    mean_projection_axis(device, src, dst, 2)
}

pub fn x_position_of_maximum_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    position_projection(
        device,
        src,
        dst,
        0,
        "x_position_of_maximum_x_projection",
        include_str!("../../kernels/x_position_of_maximum_x_projection.cl"),
    )
}

pub fn x_position_of_minimum_x_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    position_projection(
        device,
        src,
        dst,
        0,
        "x_position_of_minimum_x_projection",
        include_str!("../../kernels/x_position_of_minimum_x_projection.cl"),
    )
}

pub fn y_position_of_maximum_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    position_projection(
        device,
        src,
        dst,
        1,
        "y_position_of_maximum_y_projection",
        include_str!("../../kernels/y_position_of_maximum_y_projection.cl"),
    )
}

pub fn y_position_of_minimum_y_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    position_projection(
        device,
        src,
        dst,
        1,
        "y_position_of_minimum_y_projection",
        include_str!("../../kernels/y_position_of_minimum_y_projection.cl"),
    )
}

pub fn z_position_of_maximum_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    position_projection(
        device,
        src,
        dst,
        2,
        "z_position_of_maximum_z_projection",
        include_str!("../../kernels/z_position_of_maximum_z_projection.cl"),
    )
}

pub fn z_position_of_minimum_z_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    position_projection(
        device,
        src,
        dst,
        2,
        "z_position_of_minimum_z_projection",
        include_str!("../../kernels/z_position_of_minimum_z_projection.cl"),
    )
}

pub fn z_position_projection(
    device: &DeviceArc,
    src: &ArrayPtr,
    position: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_xy(src, dst, DType::Unknown, device)?;
    let global = {
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
        global,
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}
