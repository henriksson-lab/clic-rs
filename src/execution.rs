/// Kernel execution engine — mirrors CLIc's `execution.cpp`.
///
/// The key piece is `generate_defines()` which builds the `#define` preamble
/// that the CLIJ kernels expect before they can be compiled.
use std::collections::HashSet;

use crate::array::ArrayPtr;
use crate::backend::KernelArg;
use crate::backend_manager::BackendManager;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::types::{DType, MType};
use crate::utils::shape_to_dimension;

// ── Parameter types ───────────────────────────────────────────────────────────

pub enum ParameterValue {
    Array(ArrayPtr),
    Float(f32),
    Int(i32),
    Uint(u32),
    SizeT(usize),
}

pub type ParameterList<'a> = Vec<(&'a str, ParameterValue)>;
pub type ConstantList<'a> = Vec<(&'a str, ConstantValue)>;

pub enum ConstantValue {
    Int(i32),
    Float(f32),
    Str(String),
}

impl std::fmt::Display for ConstantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstantValue::Int(v) => write!(f, "{}", v),
            ConstantValue::Float(v) => write!(f, "{}", v),
            ConstantValue::Str(s) => write!(f, "{}", s),
        }
    }
}

pub type KernelInfo<'a> = (&'a str, &'a str); // (name, source)

// ── #define preamble generation ───────────────────────────────────────────────

/// Generate the `#define` preamble that CLIJ kernels expect, exactly matching
/// CLIc's `generateDefines()` in `execution.cpp`.
pub fn generate_defines(params: &[(&str, ParameterValue)], constants: &[(&str, ConstantValue)]) -> String {
    let mut out = String::with_capacity(4096);

    // 1. User-specified constants (e.g. `#define OP(x) fabs(x)`)
    for (key, val) in constants {
        out.push_str(&format!("#define {} {}\n", key, val));
    }
    out.push('\n');

    // Image-size getter macros
    out.push_str("#define GET_IMAGE_WIDTH(image_key) IMAGE_SIZE_ ## image_key ## _WIDTH\n");
    out.push_str("#define GET_IMAGE_HEIGHT(image_key) IMAGE_SIZE_ ## image_key ## _HEIGHT\n");
    out.push_str("#define GET_IMAGE_DEPTH(image_key) IMAGE_SIZE_ ## image_key ## _DEPTH\n");
    out.push('\n');

    // 2. Per-array defines
    let mut used_dtypes: HashSet<DType> = HashSet::new();
    let mut used_dims: HashSet<usize> = HashSet::new();

    for (key, val) in params {
        let arr_ptr = match val {
            ParameterValue::Array(a) => a,
            _ => continue,
        };
        let arr = arr_ptr.lock().unwrap();
        used_dtypes.insert(arr.dtype());
        used_dims.insert(arr.dim());

        let dim = shape_to_dimension(arr.width(), arr.height(), arr.depth());
        let dtype = arr.dtype();
        let mtype = arr.mtype();

        // Buffer path (the only path we support — IMAGE not yet implemented)
        buffer_defines(&mut out, key, dtype, mtype, dim, arr.width(), arr.height(), arr.depth());
    }

    // 3. USE_<DTYPE> defines for all unique dtypes used
    for dtype in &used_dtypes {
        out.push('\n');
        out.push_str(dtype.dimension_define());
    }

    // 4. USE_<DIM>D defines for all unique dimensions used
    for &dim in &used_dims {
        out.push('\n');
        match dim {
            1 => out.push_str("#define USE_1D"),
            2 => out.push_str("#define USE_2D"),
            3 => out.push_str("#define USE_3D"),
            _ => {}
        }
    }
    out.push_str("\n\n");
    out
}

/// Generate buffer-mode defines for a single array parameter.
/// Mirrors `bufferDefines()` in CLIc's `execution.cpp`.
fn buffer_defines(
    out: &mut String,
    key: &str,
    dtype: DType,
    _mtype: MType,
    dim: usize,
    width: usize,
    height: usize,
    depth: usize,
) {
    let ndim_strs = ["1", "2", "3"];
    let pos_type_strs = ["int", "int2", "int4"];
    let pos_strs = ["(pos0)", "(pos0, pos1)", "(pos0, pos1, pos2, 0)"];

    let idx = dim - 1;
    let ndim = ndim_strs[idx];
    let pos_type = pos_type_strs[idx];
    let pos = pos_strs[idx];
    let stype = dtype.to_short_str();
    let otype = dtype.to_ocl_str();

    // CONVERT, PIXEL_TYPE, POS_TYPE, POS_INSTANCE macros
    out.push_str(&format!("\n#define CONVERT_{key}_PIXEL_TYPE clij_convert_{otype}_sat"));
    out.push_str(&format!("\n#define IMAGE_{key}_PIXEL_TYPE {otype}"));
    out.push_str(&format!("\n#define POS_{key}_TYPE {pos_type}"));
    out.push_str(&format!("\n#define POS_{key}_INSTANCE(pos0,pos1,pos2,pos3) ({pos_type}){pos}"));
    out.push('\n');

    // Buffer-specific: type and read/write macros
    out.push_str(&format!("\n#define IMAGE_{key}_TYPE __global {otype}*"));
    out.push_str(&format!(
        "\n#define READ_{key}_IMAGE(a,b,c) read_buffer{ndim}d{stype}(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)"
    ));
    out.push_str(&format!(
        "\n#define WRITE_{key}_IMAGE(a,b,c) write_buffer{ndim}d{stype}(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)"
    ));

    // Dimension defines
    out.push_str(&format!("\n\n#define IMAGE_SIZE_{key}_WIDTH {width}"));
    out.push_str(&format!("\n#define IMAGE_SIZE_{key}_HEIGHT {height}"));
    out.push_str(&format!("\n#define IMAGE_SIZE_{key}_DEPTH {depth}"));
    out.push_str("\n\n");
}

// ── Main execute function ─────────────────────────────────────────────────────

/// Execute a CLIJ-style OpenCL kernel. Mirrors CLIc's `execute()`.
pub fn execute(
    device: &DeviceArc,
    kernel: KernelInfo,
    params: &[(&str, ParameterValue)],
    global_range: [usize; 3],
    local_range: [usize; 3],
    constants: &[(&str, ConstantValue)],
) -> Result<()> {
    let (kernel_name, kernel_source) = kernel;

    // Build full program source: defines + preamble + kernel
    let defines = generate_defines(params, constants);
    let preamble = BackendManager::get().backend().preamble().to_string();
    let program_source = format!("{}{}{}", defines, preamble, kernel_source);

    // Marshal parameters into KernelArg list (in param order)
    let mut args: Vec<KernelArg> = Vec::with_capacity(params.len());
    for (_, val) in params {
        match val {
            ParameterValue::Array(a) => {
                let lock = a.lock().unwrap();
                let mem = lock.mem_ptr().ok_or(crate::error::CleError::NotAllocated)?;
                args.push(KernelArg::Mem(mem.clone()));
            }
            ParameterValue::Float(v) => args.push(KernelArg::Float(*v)),
            ParameterValue::Int(v) => args.push(KernelArg::Int(*v)),
            ParameterValue::Uint(v) => args.push(KernelArg::Uint(*v)),
            ParameterValue::SizeT(v) => args.push(KernelArg::SizeT(*v)),
        }
    }

    BackendManager::get()
        .backend()
        .execute_kernel(device, &program_source, kernel_name, global_range, local_range, &args)
}

/// Execute a separable kernel (e.g. Gaussian blur) along each axis in turn.
/// Mirrors CLIc's `execute_separable()`.
pub fn execute_separable(
    device: &DeviceArc,
    kernel: KernelInfo,
    src: &ArrayPtr,
    dst: &ArrayPtr,
    sigma: [f32; 3],
    radius: [i32; 3],
    orders: [i32; 3],
) -> Result<()> {
    let (w, h, d) = {
        let lock = dst.lock().unwrap();
        (lock.width(), lock.height(), lock.depth())
    };
    let global = [w, h, d];

    // Allocate two temporaries
    let tmp1 = crate::array::Array::create_like(dst, device)?;
    let tmp2 = crate::array::Array::create_like(dst, device)?;

    let execute_if_needed = |dim: usize, idx: usize, input: &ArrayPtr, output: &ArrayPtr| -> Result<()> {
        if dim > 1 && sigma[idx] > 0.0 {
            let params = vec![
                ("src", ParameterValue::Array(input.clone())),
                ("dst", ParameterValue::Array(output.clone())),
                ("dim", ParameterValue::Int(idx as i32)),
                ("N", ParameterValue::Int(radius[idx])),
                ("s", ParameterValue::Float(sigma[idx])),
                ("order", ParameterValue::Int(orders[idx])),
            ];
            execute(device, kernel, &params, global, [0, 0, 0], &[])
        } else {
            // Copy input → output unchanged
            input.lock().unwrap().copy_to(output)
        }
    };

    execute_if_needed(w, 0, src, &tmp1)?;
    execute_if_needed(h, 1, &tmp1, &tmp2)?;
    execute_if_needed(d, 2, &tmp2, dst)?;
    Ok(())
}

// ── Convenience macros for building param/constant lists ─────────────────────

/// Build a `ParameterList` from `("key", value)` pairs.
/// Arrays: `("key", &array_ptr)` — scalars: `("key", 1.0f32)` / `("key", 1i32)` etc.
#[macro_export]
macro_rules! params {
    [ $( ($k:expr, $v:expr) ),* $(,)? ] => {
        vec![ $( ($k, $crate::execution::ParameterValue::from_val($v)) ),* ]
    }
}

/// Build a `ConstantList` from `("KEY", value)` pairs.
#[macro_export]
macro_rules! consts {
    [ $( ($k:expr, $v:expr) ),* $(,)? ] => {
        vec![ $( ($k, $crate::execution::ConstantValue::from_val($v)) ),* ]
    }
}

impl ParameterValue {
    pub fn from_val<T: IntoParamValue>(v: T) -> Self {
        v.into_param()
    }
}

impl ConstantValue {
    pub fn from_val<T: IntoConstValue>(v: T) -> Self {
        v.into_const()
    }
}

pub trait IntoParamValue { fn into_param(self) -> ParameterValue; }
impl IntoParamValue for ArrayPtr { fn into_param(self) -> ParameterValue { ParameterValue::Array(self) } }
impl IntoParamValue for &ArrayPtr { fn into_param(self) -> ParameterValue { ParameterValue::Array(self.clone()) } }
impl IntoParamValue for f32 { fn into_param(self) -> ParameterValue { ParameterValue::Float(self) } }
impl IntoParamValue for i32 { fn into_param(self) -> ParameterValue { ParameterValue::Int(self) } }
impl IntoParamValue for u32 { fn into_param(self) -> ParameterValue { ParameterValue::Uint(self) } }
impl IntoParamValue for usize { fn into_param(self) -> ParameterValue { ParameterValue::SizeT(self) } }

pub trait IntoConstValue { fn into_const(self) -> ConstantValue; }
impl IntoConstValue for i32 { fn into_const(self) -> ConstantValue { ConstantValue::Int(self) } }
impl IntoConstValue for f32 { fn into_const(self) -> ConstantValue { ConstantValue::Float(self) } }
impl IntoConstValue for &str { fn into_const(self) -> ConstantValue { ConstantValue::Str(self.to_string()) } }
impl IntoConstValue for String { fn into_const(self) -> ConstantValue { ConstantValue::Str(self) } }

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that generate_defines produces the correct structure without needing a GPU.
    #[test]
    fn generate_defines_no_gpu() {
        let constants = vec![("OP(x)", ConstantValue::Str("fabs(x)".into()))];
        let s = generate_defines(&[], &constants);
        assert!(s.contains("#define OP(x) fabs(x)"));
        assert!(s.contains("GET_IMAGE_WIDTH"));
    }
}
