use crate::array::{Array, ArrayPtr};
use crate::error::{CleError, Result};
use crate::execution::{native_execute, ParameterValue};
use crate::types::to_string as dtype_to_string;

const DTYPE_TOKEN: &str = "DTYPE_PLACEHOLDER";
const SLICE_STRIDED_KERNEL: &str = r#"
__kernel void slice_strided_kernel(
    __global const DTYPE_PLACEHOLDER * src,
    __global       DTYPE_PLACEHOLDER * dst,
    const int src_w,
    const int src_h,
    const int src_d,
    const int dst_w,
    const int dst_h,
    const int dst_d,
    const int x_start, const int x_step,
    const int y_start, const int y_step,
    const int z_start, const int z_step)
{
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int oz = get_global_id(2);

    if (ox >= dst_w || oy >= dst_h || oz >= dst_d)
        return;

    const int sx = x_start + ox * x_step;
    const int sy = y_start + oy * y_step;
    const int sz = z_start + oz * z_step;

    if (sx < 0 || sx >= src_w || sy < 0 || sy >= src_h || sz < 0 || sz >= src_d)
        return;

    const long src_offset = ((long)sz * src_h * src_w + (long)sy * src_w + (long)sx);
    const long dst_offset = ((long)oz * dst_h * dst_w + (long)oy * dst_w + (long)ox);

    dst[dst_offset] = src[src_offset];
}
"#;

const PASTE_STRIDED_KERNEL: &str = r#"
__kernel void paste_strided_kernel(
    __global const DTYPE_PLACEHOLDER * src,
    __global       DTYPE_PLACEHOLDER * dst,
    const int src_w,
    const int src_h,
    const int src_d,
    const int dst_w,
    const int dst_h,
    const int dst_d,
    const int x_start, const int x_step,
    const int y_start, const int y_step,
    const int z_start, const int z_step)
{
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);

    if (ix >= src_w || iy >= src_h || iz >= src_d)
        return;

    const int dx = x_start + ix * x_step;
    const int dy = y_start + iy * y_step;
    const int dz = z_start + iz * z_step;

    if (dx < 0 || dx >= dst_w || dy < 0 || dy >= dst_h || dz < 0 || dz >= dst_d)
        return;

    const long src_offset = ((long)iz * src_h * src_w + (long)iy * src_w + (long)ix);
    const long dst_offset = ((long)dz * dst_h * dst_w + (long)dy * dst_w + (long)dx);

    dst[dst_offset] = src[src_offset];
}
"#;

/// Single-axis slice specification, analogous to Python's `slice(start, stop, step)`.
///
/// Mirrors CLIc's `Slice` helper from `slicing.hpp`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slice {
    pub start: Option<i32>,
    pub stop: Option<i32>,
    pub step: i32,
    pub is_index: bool,
}

impl Slice {
    /// Full axis `[:]`.
    pub fn all() -> Self {
        Self {
            start: None,
            stop: None,
            step: 1,
            is_index: false,
        }
    }

    /// Single index `[i]`, collapsing this axis.
    pub fn index(index: i32) -> Self {
        Self {
            start: Some(index),
            stop: None,
            step: 1,
            is_index: true,
        }
    }

    /// Range `[start:stop]`.
    pub fn range(start: Option<i32>, stop: Option<i32>) -> Self {
        Self {
            start,
            stop,
            step: 1,
            is_index: false,
        }
    }

    /// Range with step `[start:stop:step]`.
    pub fn range_step(start: Option<i32>, stop: Option<i32>, step: i32) -> Self {
        Self {
            start,
            stop,
            step,
            is_index: false,
        }
    }

    /// Resolve this slice against an axis length using CPython slice semantics.
    pub fn resolve(&self, axis_len: usize) -> Result<(i32, i32, i32)> {
        let len = axis_len as i32;
        if self.step == 0 {
            return Err(CleError::Other("Slice step cannot be zero.".to_string()));
        }

        if self.is_index {
            let original = self.start.unwrap_or(0);
            let mut index = original;
            if index < 0 {
                index += len;
            }
            if index < 0 || index >= len {
                return Err(CleError::Other(format!(
                    "Slice index {original} is out of range for axis of length {len}"
                )));
            }
            return Ok((index, index + 1, 1));
        }

        let step = self.step;
        let (lower, upper) = if step > 0 { (0, len) } else { (-1, len - 1) };

        let start = match self.start {
            Some(mut start) => {
                if start < 0 {
                    start += len;
                }
                start.clamp(lower, upper)
            }
            None => {
                if step > 0 {
                    lower
                } else {
                    upper
                }
            }
        };

        let stop = match self.stop {
            Some(mut stop) => {
                if stop < 0 {
                    stop += len;
                }
                stop.clamp(lower, upper)
            }
            None => {
                if step > 0 {
                    upper
                } else {
                    lower
                }
            }
        };

        Ok((start, stop, step))
    }

    /// Number of selected elements along an axis of the given length.
    pub fn output_length(&self, axis_len: usize) -> Result<usize> {
        let (start, stop, step) = self.resolve(axis_len)?;
        if step > 0 {
            if stop <= start {
                Ok(0)
            } else {
                Ok(((stop - start + step - 1) / step) as usize)
            }
        } else if start <= stop {
            Ok(0)
        } else {
            Ok(((start - stop - step - 1) / -step) as usize)
        }
    }
}

/// Convenience constructor equivalent to CLIc's `S_()`.
pub fn s_all() -> Slice {
    Slice::all()
}

/// Convenience constructor equivalent to CLIc's `S_(index)`.
pub fn s_index(index: i32) -> Slice {
    Slice::index(index)
}

/// Convenience constructor equivalent to CLIc's `S_(start, stop)`.
pub fn s_range(start: Option<i32>, stop: Option<i32>) -> Slice {
    Slice::range(start, stop)
}

/// Convenience constructor equivalent to CLIc's `S_(start, stop, step)`.
pub fn s_range_step(start: Option<i32>, stop: Option<i32>, step: i32) -> Slice {
    Slice::range_step(start, stop, step)
}

#[derive(Clone, Copy)]
struct ResolvedSlice {
    start: i32,
    #[allow(dead_code)]
    stop: i32,
    step: i32,
    length: usize,
    is_index: bool,
}

fn resolve_slices(arr: &ArrayPtr, slices: &[Slice]) -> Result<[ResolvedSlice; 3]> {
    let shape = {
        let arr = arr.lock().unwrap();
        [arr.width, arr.height, arr.depth]
    };
    let mut resolved = [ResolvedSlice {
        start: 0,
        stop: 0,
        step: 1,
        length: 0,
        is_index: false,
    }; 3];
    for axis in 0..3 {
        let slice = slices.get(axis).cloned().unwrap_or_else(Slice::all);
        let (start, stop, step) = slice.resolve(shape[axis])?;
        let length = slice.output_length(shape[axis])?;
        resolved[axis] = ResolvedSlice {
            start,
            stop,
            step,
            length,
            is_index: slice.is_index,
        };
    }
    Ok(resolved)
}

#[allow(dead_code)]
fn replace_dtype(kernel_src: &str, dtype_str: &str) -> String {
    let mut code = kernel_src.to_string();
    let mut pos = 0;
    while let Some(found) = code[pos..].find(DTYPE_TOKEN) {
        let start = pos + found;
        let end = start + DTYPE_TOKEN.len();
        code.replace_range(start..end, dtype_str);
        pos = start + dtype_str.len();
    }
    code
}

fn compute_output_dim(resolved: &[ResolvedSlice; 3]) -> usize {
    let mut dim = 3usize;
    for item in resolved {
        if item.is_index {
            dim -= 1;
        }
    }
    dim.max(1)
}

fn is_contiguous(resolved: &[ResolvedSlice; 3]) -> bool {
    for item in resolved {
        if item.step != 1 {
            return false;
        }
    }
    true
}

fn output_shape(resolved: &[ResolvedSlice; 3]) -> [usize; 3] {
    [
        if resolved[0].is_index {
            1
        } else {
            resolved[0].length
        },
        if resolved[1].is_index {
            1
        } else {
            resolved[1].length
        },
        if resolved[2].is_index {
            1
        } else {
            resolved[2].length
        },
    ]
}

fn slice_contiguous(src: &ArrayPtr, resolved: &[ResolvedSlice; 3]) -> Result<ArrayPtr> {
    let src_origin = [
        resolved[0].start as usize,
        resolved[1].start as usize,
        resolved[2].start as usize,
    ];
    let region = [resolved[0].length, resolved[1].length, resolved[2].length];
    let [dst_width, dst_height, dst_depth] = output_shape(resolved);
    let dst_dim = compute_output_dim(resolved);
    let (dtype, mtype, device) = {
        let src = src.lock().unwrap();
        (src.dtype, src.mtype, src.device.clone())
    };
    let dst = Array::create(
        dst_width, dst_height, dst_depth, dst_dim, dtype, mtype, &device,
    )?;
    dst.lock().unwrap().allocate()?;

    let dst_origin = [0, 0, 0];
    src.lock()
        .unwrap()
        .copy_to_region(&dst, region, src_origin, dst_origin)?;
    Ok(dst)
}

fn slice_strided(src: &ArrayPtr, resolved: &[ResolvedSlice; 3]) -> Result<ArrayPtr> {
    let [dst_width, dst_height, dst_depth] = output_shape(resolved);
    let dst_dim = compute_output_dim(resolved);
    let (src_width, src_height, src_depth, dtype, mtype, device) = {
        let src = src.lock().unwrap();
        (
            src.width,
            src.height,
            src.depth,
            src.dtype,
            src.mtype,
            src.device.clone(),
        )
    };
    let dst = Array::create(
        dst_width, dst_height, dst_depth, dst_dim, dtype, mtype, &device,
    )?;
    dst.lock().unwrap().allocate()?;

    let kernel_code = replace_dtype(SLICE_STRIDED_KERNEL, dtype_to_string(dtype));
    let parameters = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("src_w", ParameterValue::Int(src_width as i32)),
        ("src_h", ParameterValue::Int(src_height as i32)),
        ("src_d", ParameterValue::Int(src_depth as i32)),
        ("dst_w", ParameterValue::Int(dst_width as i32)),
        ("dst_h", ParameterValue::Int(dst_height as i32)),
        ("dst_d", ParameterValue::Int(dst_depth as i32)),
        ("x_start", ParameterValue::Int(resolved[0].start)),
        ("x_step", ParameterValue::Int(resolved[0].step)),
        ("y_start", ParameterValue::Int(resolved[1].start)),
        ("y_step", ParameterValue::Int(resolved[1].step)),
        ("z_start", ParameterValue::Int(resolved[2].start)),
        ("z_step", ParameterValue::Int(resolved[2].step)),
    ];

    native_execute(
        &device,
        ("slice_strided_kernel", &kernel_code),
        &parameters,
        [dst_width, dst_height, dst_depth],
        [1, 1, 1],
    )?;
    Ok(dst)
}

/// Slice an array according to up to three axis slice specifications.
///
/// Mirrors CLIc's `slice` overload taking `std::vector<Slice>`.
pub fn slice(src: &ArrayPtr, slices: &[Slice]) -> Result<ArrayPtr> {
    if slices.len() > 3 {
        return Err(CleError::Other(
            "slice(): at most 3 slice specs (x, y, z) are supported.".to_string(),
        ));
    }

    let resolved = resolve_slices(src, slices)?;

    for axis in 0..3 {
        if resolved[axis].length == 0 {
            return Err(CleError::Other(format!(
                "slice(): axis {axis} produces an empty selection."
            )));
        }
    }

    if is_contiguous(&resolved) {
        return slice_contiguous(src, &resolved);
    } else {
        slice_strided(src, &resolved)
    }
}

/// Slice an array with explicit x/y/z slice arguments.
pub fn slice_xyz(
    src: &ArrayPtr,
    x_slice: Slice,
    y_slice: Slice,
    z_slice: Slice,
) -> Result<ArrayPtr> {
    slice(src, &[x_slice, y_slice, z_slice])
}

fn validate_paste_shape(src: &ArrayPtr, resolved: &[ResolvedSlice; 3]) -> Result<()> {
    let [exp_width, exp_height, exp_depth] = output_shape(resolved);
    let src_shape = {
        let src = src.lock().unwrap();
        [src.width, src.height, src.depth]
    };
    if src_shape[0] != exp_width || src_shape[1] != exp_height || src_shape[2] != exp_depth {
        return Err(CleError::Other(format!(
            "paste(): source shape ({}, {}, {}) does not match the target region ({}, {}, {}).",
            src_shape[0], src_shape[1], src_shape[2], exp_width, exp_height, exp_depth
        )));
    }
    Ok(())
}

fn paste_contiguous(src: &ArrayPtr, dst: &ArrayPtr, resolved: &[ResolvedSlice; 3]) -> Result<()> {
    let region = [resolved[0].length, resolved[1].length, resolved[2].length];
    let src_origin = [0, 0, 0];
    let dst_origin = [
        resolved[0].start as usize,
        resolved[1].start as usize,
        resolved[2].start as usize,
    ];
    src.lock()
        .unwrap()
        .copy_to_region(dst, region, src_origin, dst_origin)
}

fn paste_strided(src: &ArrayPtr, dst: &ArrayPtr, resolved: &[ResolvedSlice; 3]) -> Result<()> {
    let dtype = dst.lock().unwrap().dtype;
    let kernel_code = replace_dtype(PASTE_STRIDED_KERNEL, dtype_to_string(dtype));

    let (src_width, src_height, src_depth) = {
        let src = src.lock().unwrap();
        (src.width, src.height, src.depth)
    };
    let (dst_width, dst_height, dst_depth, device) = {
        let dst = dst.lock().unwrap();
        (dst.width, dst.height, dst.depth, dst.device.clone())
    };
    let parameters = vec![
        ("src", ParameterValue::Array(src.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("src_w", ParameterValue::Int(src_width as i32)),
        ("src_h", ParameterValue::Int(src_height as i32)),
        ("src_d", ParameterValue::Int(src_depth as i32)),
        ("dst_w", ParameterValue::Int(dst_width as i32)),
        ("dst_h", ParameterValue::Int(dst_height as i32)),
        ("dst_d", ParameterValue::Int(dst_depth as i32)),
        ("x_start", ParameterValue::Int(resolved[0].start)),
        ("x_step", ParameterValue::Int(resolved[0].step)),
        ("y_start", ParameterValue::Int(resolved[1].start)),
        ("y_step", ParameterValue::Int(resolved[1].step)),
        ("z_start", ParameterValue::Int(resolved[2].start)),
        ("z_step", ParameterValue::Int(resolved[2].step)),
    ];

    native_execute(
        &device,
        ("paste_strided_kernel", &kernel_code),
        &parameters,
        [src_width, src_height, src_depth],
        [1, 1, 1],
    )
}

/// Paste `src` into `dst` according to up to three axis slice specifications.
///
/// Mirrors CLIc's `paste` overload taking `std::vector<Slice>`.
pub fn paste(src: &ArrayPtr, dst: &ArrayPtr, slices: &[Slice]) -> Result<()> {
    if slices.len() > 3 {
        return Err(CleError::Other(
            "paste(): at most 3 slice specs (x, y, z) are supported.".to_string(),
        ));
    }

    let src_dtype = src.lock().unwrap().dtype;
    let dst_dtype = dst.lock().unwrap().dtype;
    if src_dtype != dst_dtype {
        return Err(CleError::Other(format!(
            "paste(): source and destination must have the same dtype ({} vs {}).",
            dtype_to_string(src_dtype),
            dtype_to_string(dst_dtype)
        )));
    }

    let resolved = resolve_slices(dst, slices)?;

    for axis in 0..3 {
        if resolved[axis].length == 0 {
            return Err(CleError::Other(format!(
                "paste(): axis {axis} produces an empty selection."
            )));
        }
    }

    validate_paste_shape(src, &resolved)?;
    if is_contiguous(&resolved) {
        paste_contiguous(src, dst, &resolved)
    } else {
        paste_strided(src, dst, &resolved)
    }
}

/// Paste `src` into `dst` with explicit x/y/z slice arguments.
pub fn paste_xyz(
    src: &ArrayPtr,
    dst: &ArrayPtr,
    x_slice: Slice,
    y_slice: Slice,
    z_slice: Slice,
) -> Result<()> {
    paste(src, dst, &[x_slice, y_slice, z_slice])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_resolve_matches_python_style_positive_step() {
        let slice = Slice::range_step(Some(-4), None, 2);
        assert_eq!(slice.resolve(10).unwrap(), (6, 10, 2));
        assert_eq!(slice.output_length(10).unwrap(), 2);
    }

    #[test]
    fn slice_resolve_matches_python_style_negative_step() {
        let slice = Slice::range_step(None, None, -1);
        assert_eq!(slice.resolve(4).unwrap(), (3, -1, -1));
        assert_eq!(slice.output_length(4).unwrap(), 4);
    }

    #[test]
    fn output_shape_collapses_index_axes() {
        let resolved = [
            ResolvedSlice {
                start: 1,
                stop: 2,
                step: 1,
                length: 1,
                is_index: true,
            },
            ResolvedSlice {
                start: 0,
                stop: 5,
                step: 1,
                length: 5,
                is_index: false,
            },
            ResolvedSlice {
                start: 0,
                stop: 3,
                step: 1,
                length: 3,
                is_index: false,
            },
        ];
        assert_eq!(output_shape(&resolved), [1, 5, 3]);
        assert_eq!(compute_output_dim(&resolved), 2);
    }

    #[test]
    fn replace_dtype_replaces_all_placeholders() {
        let code = "__global DTYPE_PLACEHOLDER* dst = (DTYPE_PLACEHOLDER*)0;";
        assert_eq!(
            replace_dtype(code, "float"),
            "__global float* dst = (float*)0;"
        );
    }

    #[test]
    fn is_contiguous_requires_unit_steps() {
        let mut resolved = [
            ResolvedSlice {
                start: 0,
                stop: 2,
                step: 1,
                length: 2,
                is_index: false,
            },
            ResolvedSlice {
                start: 0,
                stop: 3,
                step: 1,
                length: 3,
                is_index: false,
            },
            ResolvedSlice {
                start: 0,
                stop: 1,
                step: 1,
                length: 1,
                is_index: false,
            },
        ];

        assert!(is_contiguous(&resolved));
        resolved[1].step = 2;
        assert!(!is_contiguous(&resolved));
    }
}
