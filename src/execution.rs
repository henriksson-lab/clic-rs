/// Kernel execution engine — mirrors CLIc's `execution.cpp`.
///
/// The key piece is `generate_defines()` which builds the `#define` preamble
/// that the CLIJ kernels expect before they can be compiled.
use std::collections::BTreeSet;

use crate::array::ArrayPtr;
use crate::backend::KernelArg;
use crate::backend_manager::BackendManager;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::types::DType;
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
pub fn generate_defines(
    params: &[(&str, ParameterValue)],
    constants: &[(&str, ConstantValue)],
) -> String {
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
    let mut used_dtypes: BTreeSet<DType> = BTreeSet::new();
    let mut used_dims: BTreeSet<usize> = BTreeSet::new();

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

        // Buffer path (the only path we support — IMAGE not yet implemented)
        buffer_defines(
            &mut out,
            &ArrayDefineInfo {
                key,
                dtype,
                dim,
                width: arr.width(),
                height: arr.height(),
                depth: arr.depth(),
            },
        );
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

struct ArrayDefineInfo<'a> {
    key: &'a str,
    dtype: DType,
    dim: usize,
    width: usize,
    height: usize,
    depth: usize,
}

/// Generate buffer-mode defines for a single array parameter.
/// Mirrors `bufferDefines()` in CLIc's `execution.cpp`.
#[allow(clippy::too_many_arguments)]
fn buffer_defines(out: &mut String, info: &ArrayDefineInfo) {
    let ArrayDefineInfo {
        key,
        dtype,
        dim,
        width,
        height,
        depth,
    } = *info;
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
    out.push_str(&format!(
        "\n#define CONVERT_{key}_PIXEL_TYPE clij_convert_{otype}_sat"
    ));
    out.push_str(&format!("\n#define IMAGE_{key}_PIXEL_TYPE {otype}"));
    out.push_str(&format!("\n#define POS_{key}_TYPE {pos_type}"));
    out.push_str(&format!(
        "\n#define POS_{key}_INSTANCE(pos0,pos1,pos2,pos3) ({pos_type}){pos}"
    ));
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
    let preamble = BackendManager::get().backend().preamble();
    let mut program_source =
        String::with_capacity(defines.len() + preamble.len() + kernel_source.len());
    program_source.push_str(&defines);
    program_source.push_str(preamble);
    program_source.push_str(kernel_source);

    let args = marshal_parameters(params)?;

    BackendManager::get().backend().execute_kernel(
        device,
        &program_source,
        kernel_name,
        global_range,
        local_range,
        &args,
    )
}

/// Execute a native OpenCL kernel without CLIJ defines or preamble.
/// Mirrors CLIc's `native_execute()`.
pub fn native_execute(
    device: &DeviceArc,
    kernel: KernelInfo,
    params: &[(&str, ParameterValue)],
    global_range: [usize; 3],
    local_range: [usize; 3],
) -> Result<()> {
    let (kernel_name, kernel_source) = kernel;
    let args = marshal_parameters(params)?;

    BackendManager::get().backend().execute_kernel(
        device,
        kernel_source,
        kernel_name,
        global_range,
        local_range,
        &args,
    )
}

fn marshal_parameters(params: &[(&str, ParameterValue)]) -> Result<Vec<KernelArg>> {
    let mut args: Vec<KernelArg> = Vec::with_capacity(params.len());
    for (_, val) in params {
        match val {
            ParameterValue::Array(a) => {
                let lock = a.lock().unwrap();
                let mem = lock.mem_ptr().ok_or(CleError::NotAllocated)?;
                args.push(KernelArg::Mem(mem.clone()));
            }
            ParameterValue::Float(v) => args.push(KernelArg::Float(*v)),
            ParameterValue::Int(v) => args.push(KernelArg::Int(*v)),
            ParameterValue::Uint(v) => args.push(KernelArg::Uint(*v)),
            ParameterValue::SizeT(v) => args.push(KernelArg::SizeT(*v)),
        }
    }
    Ok(args)
}

/// Evaluate a float expression over one or more arrays into `output`.
/// Mirrors CLIc's `evaluate()` native-kernel code generation.
pub fn evaluate(
    device: &DeviceArc,
    expression: &str,
    parameters: &[ParameterValue],
    output: &ArrayPtr,
) -> Result<()> {
    if expression.trim().is_empty() {
        return Err(CleError::Other("evaluate expression is empty".to_string()));
    }

    let variable_names = extract_variable_names(expression);
    if variable_names.is_empty() {
        return Err(CleError::Other(
            "evaluate expression contains no variables".to_string(),
        ));
    }
    if variable_names.len() != parameters.len() {
        return Err(CleError::Other(format!(
            "evaluate expected {} parameters for variables {:?}, got {}",
            variable_names.len(),
            variable_names,
            parameters.len()
        )));
    }

    let (output_mem, output_dtype, total_size) = {
        let lock = output.lock().unwrap();
        let mem = lock.mem_ptr().ok_or(CleError::NotAllocated)?.clone();
        (mem, lock.dtype(), lock.size())
    };

    if total_size == 0 {
        return Err(CleError::Other("evaluate output has zero size".to_string()));
    }

    let mut arg_decls = Vec::new();
    let mut prelude = Vec::new();
    let mut args = Vec::new();
    let mut scalars = Vec::new();
    let mut has_array = false;

    for (name, param) in variable_names.iter().zip(parameters.iter()) {
        match param {
            ParameterValue::Array(array) => {
                let (mem, dtype, size) = {
                    let lock = array.lock().unwrap();
                    let mem = lock.mem_ptr().ok_or(CleError::NotAllocated)?.clone();
                    (mem, lock.dtype(), lock.size())
                };
                if size != total_size {
                    return Err(CleError::DimensionMismatch);
                }

                has_array = true;
                let array_arg = format!("_arr_{name}");
                arg_decls.push(format!(
                    "    __global const {}* {array_arg}",
                    dtype.to_ocl_str()
                ));
                prelude.push(format!(
                    "    const float {name} = (float)({array_arg}[idx]);"
                ));
                args.push(KernelArg::Mem(mem));
            }
            ParameterValue::Float(value) => scalars.push((name.as_str(), *value)),
            ParameterValue::Int(value) => scalars.push((name.as_str(), *value as f32)),
            ParameterValue::Uint(value) => scalars.push((name.as_str(), *value as f32)),
            ParameterValue::SizeT(value) => scalars.push((name.as_str(), *value as f32)),
        }
    }

    if !has_array {
        return Err(CleError::Other(
            "evaluate requires at least one array parameter".to_string(),
        ));
    }

    arg_decls.push(format!(
        "    __global {}* _arr_output",
        output_dtype.to_ocl_str()
    ));
    args.push(KernelArg::Mem(output_mem));

    for (name, value) in &scalars {
        arg_decls.push(format!("    const float {name}"));
        args.push(KernelArg::Float(*value));
    }

    if total_size > i32::MAX as usize {
        return Err(CleError::Other(
            "evaluate output is too large for native kernel size parameter".to_string(),
        ));
    }

    arg_decls.push("    const int _size".to_string());
    args.push(KernelArg::Int(total_size as i32));

    let promoted_expression = promote_builtins_to_float(expression);
    let mut kernel_source = String::new();
    kernel_source.push_str("__kernel void evaluate_kernel(\n");
    kernel_source.push_str(&arg_decls.join(",\n"));
    kernel_source.push_str("\n) {\n");
    kernel_source.push_str("    const int idx = get_global_id(0);\n");
    kernel_source.push_str("    if (idx >= _size) { return; }\n");
    for line in &prelude {
        kernel_source.push_str(line);
        kernel_source.push('\n');
    }
    kernel_source.push_str(&format!(
        "    _arr_output[idx] = ({})({});\n",
        output_dtype.to_ocl_str(),
        promoted_expression
    ));
    kernel_source.push_str("}\n");

    BackendManager::get().backend().execute_kernel(
        device,
        &kernel_source,
        "evaluate_kernel",
        [total_size, 1, 1],
        [0, 0, 0],
        &args,
    )
}

fn extract_variable_names(expression: &str) -> Vec<String> {
    let bytes = expression.as_bytes();
    let mut out = Vec::new();
    let mut idx = 0;

    while idx < bytes.len() {
        let ch = bytes[idx] as char;
        if ch.is_ascii_digit()
            || (ch == '.' && idx + 1 < bytes.len() && (bytes[idx + 1] as char).is_ascii_digit())
        {
            idx = skip_number_literal(bytes, idx);
            continue;
        }

        if is_identifier_start(ch) {
            let start = idx;
            idx += 1;
            while idx < bytes.len() && is_identifier_continue(bytes[idx] as char) {
                idx += 1;
            }
            let ident = &expression[start..idx];
            if !is_evaluate_builtin(ident) && !out.iter().any(|seen| seen == ident) {
                out.push(ident.to_string());
            }
            continue;
        }

        idx += 1;
    }

    out
}

fn skip_number_literal(bytes: &[u8], mut idx: usize) -> usize {
    if bytes[idx] == b'0' && idx + 1 < bytes.len() && matches!(bytes[idx + 1], b'x' | b'X') {
        idx += 2;
        while idx < bytes.len() && (bytes[idx] as char).is_ascii_hexdigit() {
            idx += 1;
        }
    } else {
        while idx < bytes.len() && (bytes[idx] as char).is_ascii_digit() {
            idx += 1;
        }
        if idx < bytes.len() && bytes[idx] == b'.' {
            idx += 1;
            while idx < bytes.len() && (bytes[idx] as char).is_ascii_digit() {
                idx += 1;
            }
        }
        if idx < bytes.len() && matches!(bytes[idx], b'e' | b'E') {
            idx += 1;
            if idx < bytes.len() && matches!(bytes[idx], b'+' | b'-') {
                idx += 1;
            }
            while idx < bytes.len() && (bytes[idx] as char).is_ascii_digit() {
                idx += 1;
            }
        }
    }

    while idx < bytes.len() && matches!(bytes[idx], b'f' | b'F' | b'l' | b'L' | b'u' | b'U') {
        idx += 1;
    }
    idx
}

fn promote_builtins_to_float(expression: &str) -> String {
    let bytes = expression.as_bytes();
    let mut out = String::with_capacity(expression.len());
    let mut idx = 0;

    while idx < bytes.len() {
        let ch = bytes[idx] as char;
        if is_identifier_start(ch) {
            let start = idx;
            idx += 1;
            while idx < bytes.len() && is_identifier_continue(bytes[idx] as char) {
                idx += 1;
            }
            let ident = &expression[start..idx];
            out.push_str(match ident {
                "abs" => "fabs",
                "min" => "fmin",
                "max" => "fmax",
                "power" => "pow",
                _ => ident,
            });
        } else {
            out.push(ch);
            idx += 1;
        }
    }

    out
}

fn is_identifier_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_identifier_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

fn is_evaluate_builtin(ident: &str) -> bool {
    matches!(
        ident,
        "acos"
            | "asin"
            | "atan"
            | "atan2"
            | "ceil"
            | "cos"
            | "cosh"
            | "exp"
            | "fabs"
            | "floor"
            | "fmax"
            | "fmin"
            | "fmod"
            | "log"
            | "log10"
            | "max"
            | "min"
            | "pow"
            | "power"
            | "round"
            | "sin"
            | "sinh"
            | "sqrt"
            | "tan"
            | "tanh"
            | "abs"
            | "float"
            | "double"
            | "int"
            | "uint"
            | "long"
            | "ulong"
            | "short"
            | "ushort"
            | "char"
            | "uchar"
            | "const"
            | "true"
            | "false"
    )
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

    let execute_if_needed =
        |dim: usize, idx: usize, input: &ArrayPtr, output: &ArrayPtr| -> Result<()> {
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

pub trait IntoParamValue {
    fn into_param(self) -> ParameterValue;
}
impl IntoParamValue for ArrayPtr {
    fn into_param(self) -> ParameterValue {
        ParameterValue::Array(self)
    }
}
impl IntoParamValue for &ArrayPtr {
    fn into_param(self) -> ParameterValue {
        ParameterValue::Array(self.clone())
    }
}
impl IntoParamValue for f32 {
    fn into_param(self) -> ParameterValue {
        ParameterValue::Float(self)
    }
}
impl IntoParamValue for i32 {
    fn into_param(self) -> ParameterValue {
        ParameterValue::Int(self)
    }
}
impl IntoParamValue for u32 {
    fn into_param(self) -> ParameterValue {
        ParameterValue::Uint(self)
    }
}
impl IntoParamValue for usize {
    fn into_param(self) -> ParameterValue {
        ParameterValue::SizeT(self)
    }
}

pub trait IntoConstValue {
    fn into_const(self) -> ConstantValue;
}
impl IntoConstValue for i32 {
    fn into_const(self) -> ConstantValue {
        ConstantValue::Int(self)
    }
}
impl IntoConstValue for f32 {
    fn into_const(self) -> ConstantValue {
        ConstantValue::Float(self)
    }
}
impl IntoConstValue for &str {
    fn into_const(self) -> ConstantValue {
        ConstantValue::Str(self.to_string())
    }
}
impl IntoConstValue for String {
    fn into_const(self) -> ConstantValue {
        ConstantValue::Str(self)
    }
}

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

    #[test]
    fn evaluate_variable_names_follow_first_use_order() {
        let vars = extract_variable_names("min(a, 1.0f) + pow(b, 2) + a + 0x10 + c_3");
        assert_eq!(vars, vec!["a", "b", "c_3"]);
    }

    #[test]
    fn evaluate_promotes_clic_builtins_as_tokens() {
        assert_eq!(
            promote_builtins_to_float("abs(a) + min(b, max(c, power(d, 2))) + maximum"),
            "fabs(a) + fmin(b, fmax(c, pow(d, 2))) + maximum"
        );
    }
}
