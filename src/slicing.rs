use crate::array::{pull, Array, ArrayPtr};
use crate::error::{CleError, Result};
use crate::types::{DType, GpuScalar};

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

impl Default for Slice {
    fn default() -> Self {
        Self::all()
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
    step: i32,
    length: usize,
    is_index: bool,
}

fn resolve_slices(arr: &ArrayPtr, slices: &[Slice]) -> Result<[ResolvedSlice; 3]> {
    if slices.len() > 3 {
        return Err(CleError::Other(
            "slice/paste: at most 3 slice specs (x, y, z) are supported.".to_string(),
        ));
    }
    let shape = {
        let arr = arr.lock().unwrap();
        [arr.width(), arr.height(), arr.depth()]
    };
    let mut resolved = [ResolvedSlice {
        start: 0,
        step: 1,
        length: 0,
        is_index: false,
    }; 3];
    for axis in 0..3 {
        let slice = slices.get(axis).cloned().unwrap_or_default();
        let (start, _stop, step) = slice.resolve(shape[axis])?;
        let length = slice.output_length(shape[axis])?;
        if length == 0 {
            return Err(CleError::Other(format!(
                "slice/paste: axis {axis} produces an empty selection."
            )));
        }
        resolved[axis] = ResolvedSlice {
            start,
            step,
            length,
            is_index: slice.is_index,
        };
    }
    Ok(resolved)
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

fn output_dim(resolved: &[ResolvedSlice; 3]) -> usize {
    resolved
        .iter()
        .fold(3usize, |dim, item| dim - usize::from(item.is_index))
        .max(1)
}

fn linear_index(x: usize, y: usize, z: usize, width: usize, height: usize) -> usize {
    z * width * height + y * width + x
}

fn slice_typed<T: GpuScalar>(src: &ArrayPtr, resolved: &[ResolvedSlice; 3]) -> Result<ArrayPtr> {
    let (src_width, src_height, mtype, device) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.mtype(), src.device().clone())
    };
    let [dst_width, dst_height, dst_depth] = output_shape(resolved);
    let dst = Array::create(
        dst_width,
        dst_height,
        dst_depth,
        output_dim(resolved),
        T::dtype(),
        mtype,
        &device,
    )?;
    let src_data = pull::<T>(src)?;
    let mut dst_data = Vec::with_capacity(dst_width * dst_height * dst_depth);

    for z in 0..dst_depth {
        let src_z = (resolved[2].start + z as i32 * resolved[2].step) as usize;
        for y in 0..dst_height {
            let src_y = (resolved[1].start + y as i32 * resolved[1].step) as usize;
            for x in 0..dst_width {
                let src_x = (resolved[0].start + x as i32 * resolved[0].step) as usize;
                dst_data.push(src_data[linear_index(src_x, src_y, src_z, src_width, src_height)]);
            }
        }
    }

    dst.lock().unwrap().write_from_typed(&dst_data)?;
    Ok(dst)
}

/// Slice an array according to up to three axis slice specifications.
///
/// Mirrors CLIc's `slice` overload taking `std::vector<Slice>`.
pub fn slice(src: &ArrayPtr, slices: &[Slice]) -> Result<ArrayPtr> {
    let resolved = resolve_slices(src, slices)?;
    match src.lock().unwrap().dtype() {
        DType::Float | DType::Complex => slice_typed::<f32>(src, &resolved),
        DType::Int8 => slice_typed::<i8>(src, &resolved),
        DType::Uint8 => slice_typed::<u8>(src, &resolved),
        DType::Int16 => slice_typed::<i16>(src, &resolved),
        DType::Uint16 => slice_typed::<u16>(src, &resolved),
        DType::Int32 => slice_typed::<i32>(src, &resolved),
        DType::Uint32 => slice_typed::<u32>(src, &resolved),
        DType::Unknown => Err(CleError::Other(
            "slice: unsupported dtype Unknown".to_string(),
        )),
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

fn paste_typed<T: GpuScalar>(
    src: &ArrayPtr,
    dst: &ArrayPtr,
    resolved: &[ResolvedSlice; 3],
) -> Result<()> {
    let (src_width, src_height, src_depth) = {
        let src = src.lock().unwrap();
        (src.width(), src.height(), src.depth())
    };
    let [expected_width, expected_height, expected_depth] = output_shape(resolved);
    if [src_width, src_height, src_depth] != [expected_width, expected_height, expected_depth] {
        return Err(CleError::Other(format!(
            "paste: source shape ({src_width}, {src_height}, {src_depth}) does not match target region ({expected_width}, {expected_height}, {expected_depth})."
        )));
    }

    let (dst_width, dst_height) = {
        let dst = dst.lock().unwrap();
        (dst.width(), dst.height())
    };
    let src_data = pull::<T>(src)?;
    let mut dst_data = pull::<T>(dst)?;

    for z in 0..src_depth {
        let dst_z = (resolved[2].start + z as i32 * resolved[2].step) as usize;
        for y in 0..src_height {
            let dst_y = (resolved[1].start + y as i32 * resolved[1].step) as usize;
            for x in 0..src_width {
                let dst_x = (resolved[0].start + x as i32 * resolved[0].step) as usize;
                let src_index = linear_index(x, y, z, src_width, src_height);
                let dst_index = linear_index(dst_x, dst_y, dst_z, dst_width, dst_height);
                dst_data[dst_index] = src_data[src_index];
            }
        }
    }

    dst.lock().unwrap().write_from_typed(&dst_data)?;
    Ok(())
}

/// Paste `src` into `dst` according to up to three axis slice specifications.
///
/// Mirrors CLIc's `paste` overload taking `std::vector<Slice>`.
pub fn paste(src: &ArrayPtr, dst: &ArrayPtr, slices: &[Slice]) -> Result<()> {
    let src_dtype = src.lock().unwrap().dtype();
    let dst_dtype = dst.lock().unwrap().dtype();
    if src_dtype != dst_dtype {
        return Err(CleError::Other(format!(
            "paste: source and destination must have the same dtype ({src_dtype:?} vs {dst_dtype:?})."
        )));
    }

    let resolved = resolve_slices(dst, slices)?;
    match dst_dtype {
        DType::Float | DType::Complex => paste_typed::<f32>(src, dst, &resolved),
        DType::Int8 => paste_typed::<i8>(src, dst, &resolved),
        DType::Uint8 => paste_typed::<u8>(src, dst, &resolved),
        DType::Int16 => paste_typed::<i16>(src, dst, &resolved),
        DType::Uint16 => paste_typed::<u16>(src, dst, &resolved),
        DType::Int32 => paste_typed::<i32>(src, dst, &resolved),
        DType::Uint32 => paste_typed::<u32>(src, dst, &resolved),
        DType::Unknown => Err(CleError::Other(
            "paste: unsupported dtype Unknown".to_string(),
        )),
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
                step: 1,
                length: 1,
                is_index: true,
            },
            ResolvedSlice {
                start: 0,
                step: 1,
                length: 5,
                is_index: false,
            },
            ResolvedSlice {
                start: 0,
                step: 1,
                length: 3,
                is_index: false,
            },
        ];
        assert_eq!(output_shape(&resolved), [1, 5, 3]);
        assert_eq!(output_dim(&resolved), 2);
    }
}
