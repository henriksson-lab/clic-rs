use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::INDEX;

/// Set one row to a constant value.
///
/// Mirrors CLIc's `set_row_func`.
pub fn set_row(device: &DeviceArc, src: &ArrayPtr, row_index: i32, value: f32) -> Result<ArrayPtr> {
    let params = vec![
        ("dst", ParameterValue::Array(src.clone())),
        ("dimension", ParameterValue::Int(0)),
        ("index", ParameterValue::Int(row_index)),
        ("scalar", ParameterValue::Float(value)),
    ];
    execute(
        device,
        ("set_slice", include_str!("../../kernels/set_slice.cl")),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [0, 0, 0],
        &[],
    )?;
    Ok(src.clone())
}

/// Set one column to a constant value.
///
/// Mirrors CLIc's `set_column_func`.
pub fn set_column(
    device: &DeviceArc,
    src: &ArrayPtr,
    column_index: i32,
    value: f32,
) -> Result<ArrayPtr> {
    let params = vec![
        ("dst", ParameterValue::Array(src.clone())),
        ("dimension", ParameterValue::Int(1)),
        ("index", ParameterValue::Int(column_index)),
        ("scalar", ParameterValue::Float(value)),
    ];
    execute(
        device,
        ("set_slice", include_str!("../../kernels/set_slice.cl")),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [0, 0, 0],
        &[],
    )?;
    Ok(src.clone())
}

/// Set one z-plane to a constant value.
///
/// Mirrors CLIc's `set_plane_func`.
pub fn set_plane(
    device: &DeviceArc,
    src: &ArrayPtr,
    plane_index: i32,
    value: f32,
) -> Result<ArrayPtr> {
    let params = vec![
        ("dst", ParameterValue::Array(src.clone())),
        ("dimension", ParameterValue::Int(2)),
        ("index", ParameterValue::Int(plane_index)),
        ("scalar", ParameterValue::Float(value)),
    ];
    execute(
        device,
        ("set_slice", include_str!("../../kernels/set_slice.cl")),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [0, 0, 0],
        &[],
    )?;
    Ok(src.clone())
}

/// Fill pixels with their x coordinate.
///
/// Mirrors CLIc's `set_ramp_x_func`.
pub fn set_ramp_x(device: &DeviceArc, src: &ArrayPtr) -> Result<ArrayPtr> {
    let params = vec![
        ("dst", ParameterValue::Array(src.clone())),
        ("dimension", ParameterValue::Int(0)),
    ];
    execute(
        device,
        ("set_ramp", include_str!("../../kernels/set_ramp.cl")),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [0, 0, 0],
        &[],
    )?;
    Ok(src.clone())
}

/// Fill pixels with their y coordinate.
///
/// Mirrors CLIc's `set_ramp_y_func`.
pub fn set_ramp_y(device: &DeviceArc, src: &ArrayPtr) -> Result<ArrayPtr> {
    let params = vec![
        ("dst", ParameterValue::Array(src.clone())),
        ("dimension", ParameterValue::Int(1)),
    ];
    execute(
        device,
        ("set_ramp", include_str!("../../kernels/set_ramp.cl")),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [0, 0, 0],
        &[],
    )?;
    Ok(src.clone())
}

/// Fill pixels with their z coordinate.
///
/// Mirrors CLIc's `set_ramp_z_func`.
pub fn set_ramp_z(device: &DeviceArc, src: &ArrayPtr) -> Result<ArrayPtr> {
    let params = vec![
        ("dst", ParameterValue::Array(src.clone())),
        ("dimension", ParameterValue::Int(2)),
    ];
    execute(
        device,
        ("set_ramp", include_str!("../../kernels/set_ramp.cl")),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [0, 0, 0],
        &[],
    )?;
    Ok(src.clone())
}

/// Set pixels where `x == y`.
///
/// Mirrors CLIc's `set_where_x_equals_y_func`.
pub fn set_where_x_equals_y(device: &DeviceArc, src: &ArrayPtr, value: f32) -> Result<ArrayPtr> {
    let params = vec![
        ("dst", ParameterValue::Array(src.clone())),
        ("scalar", ParameterValue::Float(value)),
    ];
    let constants = vec![(
        "COMPARISON_OP(x,y)",
        ConstantValue::Str("(x == y)".to_string()),
    )];
    execute(
        device,
        (
            "set_where_x_compare_y",
            include_str!("../../kernels/set_where_x_compare_y.cl"),
        ),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [1, 1, 1],
        &constants,
    )?;
    Ok(src.clone())
}

/// Set pixels where `x > y`.
///
/// Mirrors CLIc's `set_where_x_greater_than_y_func`.
pub fn set_where_x_greater_than_y(
    device: &DeviceArc,
    src: &ArrayPtr,
    value: f32,
) -> Result<ArrayPtr> {
    let params = vec![
        ("dst", ParameterValue::Array(src.clone())),
        ("scalar", ParameterValue::Float(value)),
    ];
    let constants = vec![(
        "COMPARISON_OP(x,y)",
        ConstantValue::Str("(x > y)".to_string()),
    )];
    execute(
        device,
        (
            "set_where_x_compare_y",
            include_str!("../../kernels/set_where_x_compare_y.cl"),
        ),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [1, 1, 1],
        &constants,
    )?;
    Ok(src.clone())
}

/// Set pixels where `x < y`.
///
/// Mirrors CLIc's `set_where_x_smaller_than_y_func`.
pub fn set_where_x_smaller_than_y(
    device: &DeviceArc,
    src: &ArrayPtr,
    value: f32,
) -> Result<ArrayPtr> {
    let params = vec![
        ("dst", ParameterValue::Array(src.clone())),
        ("scalar", ParameterValue::Float(value)),
    ];
    let constants = vec![(
        "COMPARISON_OP(x,y)",
        ConstantValue::Str("(x < y)".to_string()),
    )];
    execute(
        device,
        (
            "set_where_x_compare_y",
            include_str!("../../kernels/set_where_x_compare_y.cl"),
        ),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [1, 1, 1],
        &constants,
    )?;
    Ok(src.clone())
}

/// Write the linear pixel index plus `offset` at non-zero source pixels.
///
/// Mirrors CLIc's `set_nonzero_pixels_to_pixelindex_func`.
pub fn set_nonzero_pixels_to_pixelindex(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    offset: i32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, INDEX, device)?;
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("offset", ParameterValue::Int(offset)),
    ];
    execute(
        device,
        (
            "set_nonzero_pixels_to_pixelindex",
            include_str!("../../kernels/set_nonzero_pixels_to_pixelindex.cl"),
        ),
        &params,
        {
            let dst = dst.lock().unwrap();
            [dst.width(), dst.height(), dst.depth()]
        },
        [0, 0, 0],
        &[],
    )?;
    Ok(dst)
}

/// Set all pixels to a constant value.
///
/// Mirrors CLIc's `set_func`.
pub fn set(device: &DeviceArc, src: &ArrayPtr, scalar: f32) -> Result<ArrayPtr> {
    let params = vec![
        ("dst", ParameterValue::Array(src.clone())),
        ("scalar", ParameterValue::Float(scalar)),
    ];
    execute(
        device,
        ("set", include_str!("../../kernels/set.cl")),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [0, 0, 0],
        &[],
    )?;
    Ok(src.clone())
}

/// Set border pixels to a constant value.
///
/// Mirrors CLIc's `set_image_borders_func`.
pub fn set_image_borders(device: &DeviceArc, src: &ArrayPtr, value: f32) -> Result<ArrayPtr> {
    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("scalar", ParameterValue::Float(value)),
    ];
    execute(
        device,
        (
            "set_image_borders",
            include_str!("../../kernels/set_image_borders.cl"),
        ),
        &params,
        {
            let src = src.lock().unwrap();
            [src.width(), src.height(), src.depth()]
        },
        [0, 0, 0],
        &[],
    )?;
    Ok(src.clone())
}
