use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::DType;

fn next_power_of_2(mut v: usize) -> usize {
    if v <= 1 {
        return 1;
    }
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if usize::BITS > 32 {
        v |= v >> 32;
    }
    v + 1
}

fn select_tile_size(device: &DeviceArc, m: usize, k: usize, n: usize) -> usize {
    if device.device_type() == "cpu" || (m <= 4 && k <= 4 && n <= 4) {
        return 1;
    }

    let min_dim = m.min(k).min(n).max(1);
    let ideal = next_power_of_2((min_dim as f64).sqrt() as usize).min(32);
    for tile_size in [32_usize, 16, 8, 4, 2] {
        if tile_size <= ideal && tile_size * tile_size <= device.max_work_group_size() {
            return tile_size;
        }
    }
    1
}

fn round_up(value: usize, multiple: usize) -> usize {
    if multiple <= 1 {
        value
    } else {
        value.div_ceil(multiple) * multiple
    }
}

/// Multiply two 2D matrices, interpreting `matrix1` as MxK and `matrix2` as KxN.
pub fn multiply_matrix(
    device: &DeviceArc,
    matrix1: &ArrayPtr,
    matrix2: &ArrayPtr,
    matrix_destination: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    {
        let matrix1 = matrix1.lock().unwrap();
        let matrix2 = matrix2.lock().unwrap();
        if matrix1.dim() > 2 || matrix2.dim() > 2 {
            eprintln!(
                "Warning: multiply_matrix expected 2D arrays but got {}D and {}D.",
                matrix1.dim(),
                matrix2.dim()
            );
        }
        if matrix1.width() != matrix2.height() {
            eprintln!(
                "Warning: matrix dimensions are not compatible for multiplication. Expected (M,K)x(K,N) but got ({},{})x({},{}).",
                matrix1.height(),
                matrix1.width(),
                matrix2.height(),
                matrix2.width()
            );
        }
    }

    let (dst_width, dst_height) = {
        let matrix1 = matrix1.lock().unwrap();
        let matrix2 = matrix2.lock().unwrap();
        (matrix2.width(), matrix1.height())
    };
    let dst = tier0::create_dst(
        matrix1,
        matrix_destination,
        dst_width,
        dst_height,
        1,
        DType::Float,
        device,
    )?;
    let (m, k, n) = {
        let matrix1 = matrix1.lock().unwrap();
        let dst = dst.lock().unwrap();
        (dst.height(), matrix1.width(), dst.width())
    };
    let tile_size = select_tile_size(device, m, k, n);

    let params = vec![
        ("src0", ParameterValue::Array(matrix1.clone())),
        ("src1", ParameterValue::Array(matrix2.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![("TILE_SIZE", ConstantValue::Int(tile_size as i32))];
    execute(
        device,
        (
            "multiply_matrix",
            include_str!("../../kernels/multiply_matrix.cl"),
        ),
        &params,
        [round_up(n, tile_size), round_up(m, tile_size), 1],
        [tile_size, tile_size, 1],
        &constants,
    )?;
    Ok(dst)
}
