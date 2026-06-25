use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::execution::{execute, native_execute, ParameterValue};
use crate::tier0;
use crate::types::{to_string, DType, MType};

/// Copy src to dst.
pub fn copy(device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, DType::Unknown, device)?;
    let (src_dtype, src_mtype, dst_dtype, dst_mtype, dst_size) = {
        let src = src.lock().unwrap();
        let dst = dst.lock().unwrap();
        (
            src.dtype(),
            src.mtype(),
            dst.dtype(),
            dst.mtype(),
            dst.size(),
        )
    };

    if src_dtype == dst_dtype {
        // use built-in copy if data types are the same
        src.lock().unwrap().copy_to(&dst)?;
        return Ok(dst);
    }

    if src_mtype == MType::Image || dst_mtype == MType::Image {
        // use image copy kernel if either src or dst is an image
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
            ("copy", include_str!("../../kernels/copy.cl")),
            &params,
            range,
            [0, 0, 0],
            &[],
        )?;
        return Ok(dst);
    }

    // use basic copy kernel for the rest

    let input_type = to_string(src_dtype);
    let output_type = to_string(dst_dtype);
    let convert = if dst_dtype == DType::Float {
        "convert_float".to_string()
    } else {
        format!("convert_{output_type}_sat")
    };
    let kernel_name = "copy";

    let kernel_source = format!(
        "__kernel void copy(__global const {input_type}* src, __global {output_type}* dst, const uint n) {{ const uint gid = get_global_id(0); if (gid < n) dst[gid] = {convert}(src[gid]); }}"
    );

    let params = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("n", ParameterValue::Uint(dst_size as u32)),
    ];

    let local_item_size = 256.min(device.get_maximum_work_group_size());
    let global_item_size =
        ((dst_size as f64 / local_item_size as f64).ceil() as usize) * local_item_size;
    let global_range = [global_item_size, 1, 1];
    let local_range = [local_item_size, 1, 1];

    native_execute(
        device,
        (kernel_name, &kernel_source),
        &params,
        global_range,
        local_range,
    )?;
    Ok(dst)
}
