/// Kernel execution engine — mirrors CLIc's `execution.cpp`.
///
/// The key piece is `generate_defines()` which builds the `#define` preamble
/// that the CLIJ kernels expect before they can be compiled.
use std::collections::BTreeSet;

use crate::array::{Array, ArrayPtr};
use crate::backend::KernelArg;
use crate::backend_manager::BackendManager;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::translator::OpenCLToCUDATranslator;
use crate::types::{to_short_string, to_string as dtype_to_string, DType, MType};
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

pub type KernelInfo<'a> = (&'a str, &'a str); // (name, source)

/// Translate the OpenCL syntax used by simple CLIJ kernels into CUDA syntax.
/// Mirrors CLIc's `translateOpenclToCuda()` entry point for non-backend use.
pub fn translate_opencl_to_cuda(opencl_code: &str) -> String {
    let translator = OpenCLToCUDATranslator::new();
    translator.translate(opencl_code)
}

// ── #define preamble generation ───────────────────────────────────────────────

/// Generate the `#define` preamble that CLIJ kernels expect, exactly matching
/// CLIc's `generateDefines()` in `execution.cpp`.
pub fn generate_defines(
    params: &[(&str, ParameterValue)],
    constants: &[(&str, ConstantValue)],
    device_is_cuda: bool,
) -> String {
    let mut defines = String::with_capacity(4096);
    defines.push_str(&common_defines(constants));
    defines.push_str(&array_defines(params, device_is_cuda));
    defines
}

fn common_defines(constants: &[(&str, ConstantValue)]) -> String {
    let mut out = String::new();
    for (key, val) in constants {
        let value = match val {
            ConstantValue::Int(v) => v.to_string(),
            ConstantValue::Float(v) => v.to_string(),
            ConstantValue::Str(s) => s.clone(),
        };
        out.push_str(&format!("#define {} {}\n", key, value));
    }
    out.push('\n');

    // Image-size getter macros
    out.push_str("\n#define GET_IMAGE_WIDTH(image_key) IMAGE_SIZE_ ## image_key ## _WIDTH");
    out.push_str("\n#define GET_IMAGE_HEIGHT(image_key) IMAGE_SIZE_ ## image_key ## _HEIGHT");
    out.push_str("\n#define GET_IMAGE_DEPTH(image_key) IMAGE_SIZE_ ## image_key ## _DEPTH");
    out.push('\n');
    out
}

fn array_defines(params: &[(&str, ParameterValue)], device_is_cuda: bool) -> String {
    let mut out = String::new();
    for (key, val) in params {
        let arr_ptr = match val {
            ParameterValue::Array(a) => a,
            _ => continue,
        };
        let arr = arr_ptr.lock().unwrap();

        let dim = shape_to_dimension(arr.width(), arr.height(), arr.depth());
        if arr.mtype() == MType::Buffer || device_is_cuda {
            buffer_defines(&mut out, key, &arr, dim, device_is_cuda);
        } else {
            image_defines(&mut out, key, &arr, dim, device_is_cuda);
        }
        out.push_str(&format!(
            "\n\n#define IMAGE_SIZE_{}_WIDTH {}",
            key,
            arr.width()
        ));
        out.push_str(&format!(
            "\n#define IMAGE_SIZE_{}_HEIGHT {}",
            key,
            arr.height()
        ));
        out.push_str(&format!(
            "\n#define IMAGE_SIZE_{}_DEPTH {}",
            key,
            arr.depth()
        ));
        out.push_str("\n\n");
    }
    out
}

/// Generate buffer-mode defines for a single array parameter.
/// Mirrors `bufferDefines()` in CLIc's `execution.cpp`.
fn buffer_defines(out: &mut String, key: &str, arr: &Array, dim: usize, device_is_cuda: bool) {
    let ndim_map = ["1", "2", "3"];
    let pos_type_map = ["int", "int2", "int4"];
    let pos_map = ["(pos0)", "(pos0, pos1)", "(pos0, pos1, pos2, 0)"];

    let dim_index = dim - 1;
    let ndim = ndim_map[dim_index];
    let pos_type = pos_type_map[dim_index];
    let pos = pos_map[dim_index];
    let stype = to_short_string(arr.dtype());
    let dtype = dtype_to_string(arr.dtype());
    let access_type = if device_is_cuda { "" } else { "__global " };
    let prefix = if !device_is_cuda || pos_type == "int" {
        format!("({pos_type})")
    } else {
        format!("make_{pos_type}")
    };

    // CONVERT, PIXEL_TYPE, POS_TYPE, POS_INSTANCE macros
    out.push_str(&format!(
        "\n#define CONVERT_{key}_PIXEL_TYPE clij_convert_{dtype}_sat"
    ));
    out.push_str(&format!("\n#define IMAGE_{key}_PIXEL_TYPE {dtype}"));
    out.push_str(&format!("\n#define POS_{key}_TYPE {pos_type}"));
    out.push_str(&format!(
        "\n#define POS_{key}_INSTANCE(pos0,pos1,pos2,pos3) {prefix}{pos}"
    ));
    out.push('\n');

    // Buffer-specific: type and read/write macros
    out.push_str(&format!("\n#define IMAGE_{key}_TYPE {access_type}{dtype}*"));
    out.push_str(&format!(
        "\n#define READ_{key}_IMAGE(a,b,c) read_buffer{ndim}d{stype}(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)"
    ));
    out.push_str(&format!(
        "\n#define WRITE_{key}_IMAGE(a,b,c) write_buffer{ndim}d{stype}(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)"
    ));
}

fn image_defines(out: &mut String, key: &str, arr: &Array, dim: usize, device_is_cuda: bool) {
    let ndim_map = ["1", "2", "3"];
    let pos_int_type_map = ["int", "int2", "int4"];
    let pos_float_type_map = ["float", "float2", "float4"];
    let pos_map = ["(pos0)", "(pos0, pos1)", "(pos0, pos1, pos2, 0)"];

    let dim_index = dim - 1;
    let ndim = ndim_map[dim_index];
    let pos = pos_map[dim_index];
    let stype = to_short_string(arr.dtype());
    let dtype = dtype_to_string(arr.dtype());
    let is_output = key.contains("dst") || key.contains("destination") || key.contains("output");
    let access_type = if is_output {
        "__write_only"
    } else {
        "__read_only"
    };
    let pos_type = if is_output {
        pos_int_type_map[dim_index]
    } else {
        pos_float_type_map[dim_index]
    };
    let prefix1 = if !device_is_cuda || pos_type == "int" {
        format!("({pos_type})")
    } else {
        format!("make_{pos_type}")
    };

    out.push_str(&format!(
        "\n#define CONVERT_{key}_PIXEL_TYPE clij_convert_{dtype}_sat"
    ));
    out.push_str(&format!("\n#define IMAGE_{key}_PIXEL_TYPE {dtype}"));
    out.push_str(&format!("\n#define POS_{key}_TYPE {pos_type}"));
    out.push_str(&format!(
        "\n#define POS_{key}_INSTANCE(pos0,pos1,pos2,pos3) {prefix1}{pos}"
    ));
    out.push('\n');

    let prefix2 = match stype.as_bytes().first().copied() {
        Some(b'u') => "ui",
        Some(b'f') => "f",
        _ => "i",
    };
    let img_type_name = format!("{access_type} image{ndim}d_t");
    out.push_str(&format!("\n#define IMAGE_{key}_TYPE {img_type_name}"));
    out.push_str(&format!(
        "\n#define READ_{key}_IMAGE(a,b,c) read_image{prefix2}(a,b,c)"
    ));
    out.push_str(&format!(
        "\n#define WRITE_{key}_IMAGE(a,b,c) write_image{prefix2}(a,b,c)"
    ));
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
    let mut kernel_source = kernel_source;
    let mut kernel_preamble = BackendManager::get_instance()
        .backend()
        .get_preamble()?
        .to_string();
    let platform = device.get_platform();
    let device_is_cuda = platform == "CUDA" || platform == "NVIDIA";
    let mut defines = generate_defines(params, constants, device_is_cuda);
    platform_options(device, &mut kernel_preamble);
    let kernel_source_string;
    if device_is_cuda {
        kernel_source_string = translate_opencl_to_cuda(kernel_source);
        kernel_source = &kernel_source_string;
    }

    let mut used_dtypes: BTreeSet<DType> = BTreeSet::new();
    let mut used_dims: BTreeSet<usize> = BTreeSet::new();
    for (_, val) in params {
        if let ParameterValue::Array(arr_ptr) = val {
            let arr = arr_ptr.lock().unwrap();
            used_dtypes.insert(arr.dtype());
            used_dims.insert(arr.dim());
        }
    }
    let args = marshal_parameters(params)?;

    for dtype in used_dtypes {
        defines.push('\n');
        defines.push_str(match dtype {
            DType::Int8 => "#define USE_CHAR",
            DType::Uint8 => "#define USE_UCHAR",
            DType::Int16 => "#define USE_SHORT",
            DType::Uint16 => "#define USE_USHORT",
            DType::Int32 => "#define USE_INT",
            DType::Uint32 => "#define USE_UINT",
            DType::Float | DType::Complex => "#define USE_FLOAT",
            DType::Unknown => "",
        });
    }
    for dim in used_dims {
        defines.push('\n');
        match dim {
            1 => defines.push_str("#define USE_1D"),
            2 => defines.push_str("#define USE_2D"),
            3 => defines.push_str("#define USE_3D"),
            _ => {}
        }
    }
    defines.push_str("\n\n");
    let mut program_source =
        String::with_capacity(defines.len() + kernel_preamble.len() + kernel_source.len());
    program_source.push_str(&defines);
    program_source.push_str(&kernel_preamble);
    program_source.push_str(kernel_source);

    BackendManager::get_instance().backend().execute_kernel(
        device,
        &program_source,
        kernel_name,
        global_range,
        local_range,
        &args,
    )
}

fn platform_options(device: &DeviceArc, source: &mut String) {
    if device.get_platform().contains("AMD") {
        source.insert_str(0, "#pragma OPENCL EXTENSION cl_amd_printf : enable\n");
    }
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
    let mut kernel_source = kernel_source;
    let kernel_source_string;
    let platform = device.get_platform();
    if platform == "CUDA" || platform == "NVIDIA" {
        kernel_source_string = translate_opencl_to_cuda(kernel_source);
        kernel_source = &kernel_source_string;
    }

    let args = marshal_parameters(params)?;
    BackendManager::get_instance().backend().execute_kernel(
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
                let mem = lock.get_ptr().ok_or(CleError::NotAllocated)?;
                args.push(KernelArg::Mem(mem));
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
    if parameters.is_empty() {
        return Err(CleError::Other(
            "Error: 'parameters' list is empty in evaluate().".to_string(),
        ));
    }
    if expression.is_empty() {
        return Err(CleError::Other(
            "Error: 'expression' is empty in evaluate().".to_string(),
        ));
    }

    // Extract variable names from expression, in order of first appearance
    let var_names = extract_variable_names(expression);
    if var_names.len() != parameters.len() {
        return Err(CleError::Other(format!(
            "Error: expression has {} variable(s) but {} parameter(s) were provided.",
            var_names.len(),
            parameters.len()
        )));
    }

    // Promote integer math builtins to float equivalents (abs->fabs, min->fmin, max->fmax)
    let float_expression = promote_builtins_to_float(expression);

    // Classify each parameter as array or scalar, paired with its variable name
    struct ArrayParam {
        name: String,
        arr: ArrayPtr,
    }

    struct ScalarParam {
        name: String,
        val: f32,
    }

    let mut arrays = Vec::new();
    let mut scalars = Vec::new();

    for idx in 0..parameters.len() {
        let param = &parameters[idx];
        let name = &var_names[idx];

        match param {
            ParameterValue::Array(array) => {
                let size = {
                    let lock = array.lock().unwrap();
                    lock.get_ptr().ok_or(CleError::NotAllocated)?;
                    lock.size()
                };
                let output_size = {
                    let lock = output.lock().unwrap();
                    lock.size()
                };
                if size != output_size {
                    return Err(CleError::Other(format!(
                        "Error: array '{}' size ({}) does not match output size ({}).",
                        name, size, output_size
                    )));
                }

                arrays.push(ArrayParam {
                    name: name.clone(),
                    arr: array.clone(),
                });
            }
            ParameterValue::Float(value) => scalars.push(ScalarParam {
                name: name.clone(),
                val: *value,
            }),
            ParameterValue::Int(value) => scalars.push(ScalarParam {
                name: name.clone(),
                val: *value as f32,
            }),
            ParameterValue::Uint(value) => scalars.push(ScalarParam {
                name: name.clone(),
                val: *value as f32,
            }),
            ParameterValue::SizeT(value) => scalars.push(ScalarParam {
                name: name.clone(),
                val: *value as f32,
            }),
        }
    }

    if arrays.is_empty() {
        return Err(CleError::Other(
            "Error: at least one Array parameter is required in evaluate().".to_string(),
        ));
    }

    let addr_qualifier = "__global ";
    let kernel_keyword = "__kernel";
    let total_size = {
        let lock = output.lock().unwrap();
        lock.size()
    };

    // --- Generate pure OpenCL/CUDA 1D kernel source ---
    let mut kernel_source = String::new();
    kernel_source.push_str(kernel_keyword);
    kernel_source.push_str(" void evaluate_kernel(\n");

    // Input array parameters (typed by their actual dtype)
    for array in &arrays {
        let dtype = {
            let lock = array.arr.lock().unwrap();
            dtype_to_string(lock.dtype())
        };
        kernel_source.push_str(&format!(
            "    {addr_qualifier}const {dtype}* _arr_{},\n",
            array.name
        ));
    }

    // Output array parameter
    let output_dtype = {
        let lock = output.lock().unwrap();
        dtype_to_string(lock.dtype())
    };
    kernel_source.push_str(&format!(
        "    {addr_qualifier}{output_dtype}* _arr_output,\n"
    ));

    // Scalar parameters (all passed as float)
    for scalar in &scalars {
        kernel_source.push_str(&format!("    const float {},\n", scalar.name));
    }

    // Total number of elements
    kernel_source.push_str("    const int _size\n");
    kernel_source.push_str(") {\n");
    // 1D thread index
    kernel_source.push_str("    const int idx = get_global_id(0);\n");
    // Bounds check
    kernel_source.push_str("    if (idx >= _size) return;\n\n");

    // Read each input array element and cast to float
    for array in &arrays {
        kernel_source.push_str(&format!(
            "    const float {} = (float)_arr_{}[idx];\n",
            array.name, array.name
        ));
    }
    kernel_source.push('\n');

    // Evaluate expression (all in float) and cast result to output dtype
    kernel_source.push_str(&format!(
        "    _arr_output[idx] = ({})({});\n",
        output_dtype, float_expression
    ));
    kernel_source.push_str("}\n");

    let kernel_name = "evaluate_kernel";
    // convert OpenCL kernel to CUDA if needed
    let platform = device.get_platform();
    if platform == "CUDA" || platform == "NVIDIA" {
        kernel_source = translate_opencl_to_cuda(&kernel_source);
    }

    // --- Build argument lists ---
    let mut args = Vec::with_capacity(arrays.len() + 1 + scalars.len() + 1);
    for array in arrays {
        let mem = {
            let lock = array.arr.lock().unwrap();
            lock.get_ptr().ok_or(CleError::NotAllocated)?
        };
        args.push(KernelArg::Mem(mem));
    }
    let output_mem = {
        let lock = output.lock().unwrap();
        lock.get_ptr().ok_or(CleError::NotAllocated)?
    };
    args.push(KernelArg::Mem(output_mem));

    // Scalars (all as float)
    for scalar in scalars {
        args.push(KernelArg::Float(scalar.val));
    }

    // Total size parameter
    args.push(KernelArg::Int(total_size as i32));

    // Execute as 1D kernel with max local work group size
    let max_local = device.get_maximum_work_group_size();
    let global_size_padded = ((total_size + max_local - 1) / max_local) * max_local;

    BackendManager::get_instance().backend().execute_kernel(
        device,
        &kernel_source,
        kernel_name,
        [global_size_padded, 1, 1],
        [max_local, 1, 1],
        &args,
    )
}

fn extract_variable_names(expression: &str) -> Vec<String> {
    let builtins: BTreeSet<&'static str> = [
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "exp",
        "exp2",
        "exp10",
        "log",
        "log2",
        "log10",
        "pow",
        "power",
        "pown",
        "powr",
        "sqrt",
        "rsqrt",
        "cbrt",
        "fabs",
        "abs",
        "fmin",
        "fmax",
        "fmod",
        "remainder",
        "ceil",
        "floor",
        "round",
        "trunc",
        "rint",
        "clamp",
        "mix",
        "step",
        "smoothstep",
        "sign",
        "min",
        "max",
        "mad",
        "fma",
        "copysign",
        "fdim",
        "hypot",
        "ldexp",
        "frexp",
        "native_sin",
        "native_cos",
        "native_exp",
        "native_log",
        "native_sqrt",
        "native_tan",
        "native_recip",
        "native_rsqrt",
        "native_powr",
        "half_sin",
        "half_cos",
        "half_exp",
        "half_log",
        "half_sqrt",
        "half_tan",
        "half_recip",
        "half_rsqrt",
        "half_powr",
        "isnan",
        "isinf",
        "isfinite",
        "isnormal",
        "signbit",
        "select",
        "bitselect",
        "convert_float",
        "convert_int",
        "convert_uint",
        "convert_char",
        "convert_uchar",
        "convert_short",
        "convert_ushort",
        "as_float",
        "as_int",
        "as_uint",
        "float",
        "double",
        "half",
        "int",
        "uint",
        "char",
        "uchar",
        "short",
        "ushort",
        "long",
        "ulong",
        "bool",
        "void",
        "const",
        "unsigned",
        "return",
        "if",
        "else",
        "for",
        "while",
        "do",
        "break",
        "continue",
        "true",
        "false",
    ]
    .into_iter()
    .collect();

    let bytes = expression.as_bytes();
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut idx = 0;
    let len = bytes.len();

    while idx < len {
        let ch = bytes[idx] as char;
        if ch.is_ascii_digit()
            || (ch == '.' && idx + 1 < len && (bytes[idx + 1] as char).is_ascii_digit())
        {
            if bytes[idx] == b'0' && idx + 1 < len && matches!(bytes[idx + 1], b'x' | b'X') {
                idx += 2;
                while idx < len && (bytes[idx] as char).is_ascii_hexdigit() {
                    idx += 1;
                }
            } else {
                while idx < len && ((bytes[idx] as char).is_ascii_digit() || bytes[idx] == b'.') {
                    idx += 1;
                }
                if idx < len && matches!(bytes[idx], b'e' | b'E') {
                    idx += 1;
                    if idx < len && matches!(bytes[idx], b'+' | b'-') {
                        idx += 1;
                    }
                    while idx < len && (bytes[idx] as char).is_ascii_digit() {
                        idx += 1;
                    }
                }
            }

            while idx < len && matches!(bytes[idx], b'f' | b'F' | b'l' | b'L' | b'u' | b'U') {
                idx += 1;
            }
            continue;
        }

        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = idx;
            while idx < len && ((bytes[idx] as char).is_ascii_alphanumeric() || bytes[idx] == b'_')
            {
                idx += 1;
            }
            let ident = &expression[start..idx];
            if !builtins.contains(ident) && seen.insert(ident.to_string()) {
                out.push(ident.to_string());
            }
            continue;
        }

        idx += 1;
    }

    out
}

fn promote_builtins_to_float(expression: &str) -> String {
    let replacements = [
        ("abs", "fabs"),
        ("min", "fmin"),
        ("max", "fmax"),
        ("power", "pow"),
    ];
    let mut result = expression.to_string();

    for (from, to) in replacements {
        let mut pos = 0;
        while let Some(found) = result[pos..].find(from) {
            pos += found;
            let end = pos + from.len();
            let preceded_by_id = pos > 0
                && (result.as_bytes()[pos - 1].is_ascii_alphanumeric()
                    || result.as_bytes()[pos - 1] == b'_');
            let followed_by_id = end < result.len()
                && (result.as_bytes()[end].is_ascii_alphanumeric()
                    || result.as_bytes()[end] == b'_');

            if !preceded_by_id && !followed_by_id {
                result.replace_range(pos..end, to);
                pos += to.len();
            } else {
                pos += from.len();
            }
        }
    }

    result
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
    let tmp1 = crate::tier0::create_like(dst, None, DType::Unknown, device)?;
    let tmp2 = crate::tier0::create_like(dst, None, DType::Unknown, device)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Array, ArrayPtr};
    use crate::device::Device;
    use crate::types::MType;
    use opencl3::program::Program;
    use std::sync::{Arc, Mutex};

    struct DummyDevice;

    impl Device for DummyDevice {
        fn get_name(&self) -> &str {
            "dummy"
        }
        fn get_device_type(&self) -> &str {
            "dummy"
        }
        fn support_image(&self) -> bool {
            false
        }
        fn get_maximum_buffer_size(&self) -> usize {
            0
        }
        fn get_maximum_work_group_size(&self) -> usize {
            1
        }
        fn get_local_memory_size(&self) -> usize {
            usize::MAX
        }
        fn get_platform(&self) -> String {
            String::new()
        }
        fn finish(&self) {}
        fn get_program_from_cache(&self, _key: &str) -> Option<Arc<Program>> {
            None
        }
        fn add_program_to_cache(&self, _key: String, _program: Arc<Program>) {}
        fn device_hash(&self) -> String {
            "dummy".to_string()
        }
    }

    struct PlatformDevice(&'static str);

    impl Device for PlatformDevice {
        fn get_name(&self) -> &str {
            "platform"
        }
        fn get_device_type(&self) -> &str {
            "gpu"
        }
        fn support_image(&self) -> bool {
            false
        }
        fn get_maximum_buffer_size(&self) -> usize {
            0
        }
        fn get_maximum_work_group_size(&self) -> usize {
            1
        }
        fn get_local_memory_size(&self) -> usize {
            usize::MAX
        }
        fn get_platform(&self) -> String {
            self.0.to_string()
        }
        fn finish(&self) {}
        fn get_program_from_cache(&self, _key: &str) -> Option<Arc<Program>> {
            None
        }
        fn add_program_to_cache(&self, _key: String, _program: Arc<Program>) {}
        fn device_hash(&self) -> String {
            "platform".to_string()
        }
    }

    fn unallocated_array_with_mtype(
        width: usize,
        height: usize,
        depth: usize,
        dtype: DType,
        mtype: MType,
    ) -> ArrayPtr {
        Arc::new(Mutex::new(Array {
            width,
            height,
            depth,
            dim: shape_to_dimension(width, height, depth),
            dtype,
            mtype,
            device: Arc::new(DummyDevice),
            mem: None,
            owns_memory: true,
        }))
    }

    fn unallocated_array(width: usize, height: usize, depth: usize, dtype: DType) -> ArrayPtr {
        unallocated_array_with_mtype(width, height, depth, dtype, MType::Buffer)
    }

    /// Verify that generate_defines produces the correct structure without needing a GPU.
    #[test]
    fn generate_defines_no_gpu() {
        let constants = vec![("OP(x)", ConstantValue::Str("fabs(x)".into()))];
        let s = generate_defines(&[], &constants, false);
        assert!(s.contains("#define OP(x) fabs(x)"));
        assert!(s.contains("GET_IMAGE_WIDTH"));
    }

    #[test]
    fn generate_defines_leaves_used_type_defines_to_execute_step() {
        let src = unallocated_array(4, 3, 1, DType::Float);
        let params = vec![("src", ParameterValue::Array(src))];
        let defines = generate_defines(&params, &[], false);

        assert!(defines.contains("#define IMAGE_src_TYPE __global float*"));
        assert!(!defines.contains("#define USE_FLOAT"));
        assert!(!defines.contains("#define USE_2D"));
    }

    #[test]
    fn array_defines_cuda_uses_buffer_path_for_images() {
        let src = unallocated_array_with_mtype(4, 3, 1, DType::Float, MType::Image);
        let params = vec![("src", ParameterValue::Array(src))];
        let defines = array_defines(&params, true);

        assert!(defines.contains("#define IMAGE_src_TYPE float*"));
        assert!(defines.contains("#define POS_src_INSTANCE(pos0,pos1,pos2,pos3) make_int2"));
        assert!(!defines.contains("image2d_t"));
    }

    #[test]
    fn platform_options_prepends_amd_printf_extension() {
        let amd: DeviceArc = Arc::new(PlatformDevice("AMD Accelerated Parallel Processing"));
        let mut source = "__kernel void k() {}".to_string();
        platform_options(&amd, &mut source);
        assert!(source.starts_with("#pragma OPENCL EXTENSION cl_amd_printf : enable\n"));

        let other: DeviceArc = Arc::new(PlatformDevice("Portable Computing Language"));
        let mut source = "__kernel void k() {}".to_string();
        platform_options(&other, &mut source);
        assert_eq!(source, "__kernel void k() {}");
    }

    #[test]
    fn evaluate_variable_names_follow_first_use_order() {
        let vars = extract_variable_names("min(a, 1.0f) + pow(b, 2) + a + 0x10 + c_3");
        assert_eq!(vars, vec!["a", "b", "c_3"]);
    }

    #[test]
    fn evaluate_variable_names_skip_opencl_builtins() {
        let vars = extract_variable_names(
            "native_sin(a) + convert_float(b) + select(c, d, isfinite(e)) + half_sqrt(f)",
        );
        assert_eq!(vars, vec!["a", "b", "c", "d", "e", "f"]);
    }

    #[test]
    fn evaluate_promotes_clic_builtins_as_tokens() {
        assert_eq!(
            promote_builtins_to_float("abs(a) + min(b, max(c, power(d, 2))) + maximum + abs"),
            "fabs(a) + fmin(b, fmax(c, pow(d, 2))) + maximum + fabs"
        );
    }
}
