//! Runtime OpenCL-to-CUDA source translation.
//!
//! This mirrors CLIc's `OpenCLToCUDATranslator`: it is intentionally a
//! text-rewriting translator rather than a full OpenCL parser.

use regex::Regex;

#[derive(Debug, Default, Clone)]
pub struct OpenCLToCUDATranslator;

impl OpenCLToCUDATranslator {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    #[must_use]
    pub fn translate(&self, opencl_source: &str) -> String {
        let mut code = opencl_source.to_string();
        self.translate_in_place(&mut code);
        code
    }

    pub fn translate_in_place(&self, code: &mut String) {
        Self::translate_pragmas(code);
        Self::translate_samplers(code);
        Self::translate_image_operations(code);
        Self::translate_qualifiers(code);
        Self::translate_address_spaces(code);
        Self::translate_work_item_functions(code);
        Self::translate_synchronization(code);
        Self::translate_vector_constructors(code);
        Self::translate_vector_access(code);
        Self::translate_atomics(code);
        Self::translate_type_conversions(code);
        Self::translate_math_functions(code);
        Self::translate_misc_builtins(code);
        Self::cleanup_double_qualifiers(code);
    }

    pub fn replace_all(str_: &mut String, from: &str, to: &str) {
        if from.is_empty() {
            return;
        }
        *str_ = str_.replace(from, to);
    }

    pub fn regex_replace_all(str_: &mut String, pattern: &str, replacement: &str) {
        let pattern = Regex::new(pattern).expect("invalid OpenCL-to-CUDA regex pattern");
        *str_ = pattern.replace_all(str_, replacement).into_owned();
    }

    pub fn replace_vector_constructor(code: &mut String, opencl_ctor: &str, cuda_ctor: &str) {
        let mut pos = 0;
        while let Some(found) = code[pos..].find(opencl_ctor) {
            pos += found;
            code.replace_range(pos..pos + opencl_ctor.len(), cuda_ctor);
            pos += cuda_ctor.len();

            let mut depth = 1i32;
            let mut close = None;
            for (offset, ch) in code[pos..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(pos + offset);
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if let Some(close) = close {
                code.replace_range(close..close + 1, ")");
                pos = close + 1;
            }
        }
    }

    pub fn translate_pragmas(code: &mut String) {
        Self::replace_all(code, "#pragma OPENCL", "// #pragma OPENCL");

        let mut result = String::with_capacity(code.len());
        for line in code.lines() {
            let trimmed = line.trim_start_matches([' ', '\t']);
            if trimmed.starts_with("#pragma") && !trimmed.contains("// #pragma") {
                result.push_str("// ");
            }
            result.push_str(line);
            result.push('\n');
        }
        *code = result;

        Self::replace_all(code, "#define CL_VERSION_", "// #define CL_VERSION_");
    }

    pub fn translate_samplers(code: &mut String) {
        for (from, to) in [
            ("__constant sampler_t", "__device__ int"),
            ("const sampler_t", "__device__ int"),
            ("sampler_t", "int"),
            ("CLK_NORMALIZED_COORDS_FALSE", "0"),
            ("CLK_NORMALIZED_COORDS_TRUE", "1"),
            ("CLK_ADDRESS_CLAMP_TO_EDGE", "0"),
            ("CLK_ADDRESS_CLAMP", "0"),
            ("CLK_ADDRESS_REPEAT", "0"),
            ("CLK_ADDRESS_NONE", "0"),
            ("CLK_FILTER_NEAREST", "0"),
            ("CLK_FILTER_LINEAR", "0"),
        ] {
            Self::replace_all(code, from, to);
        }
    }

    pub fn translate_qualifiers(code: &mut String) {
        Self::replace_all(code, "__kernel void", "extern \"C\" __global__ void");
        Self::replace_all(code, "kernel void", "extern \"C\" __global__ void");
        Self::replace_all(
            code,
            "__attribute__((reqd_work_group_size(",
            "// __attribute__((reqd_work_group_size(",
        );
        Self::regex_replace_all(code, r"\binline\b", "__device__ inline");
    }

    pub fn translate_address_spaces(code: &mut String) {
        Self::regex_replace_all(code, r"__global\b", "");
        Self::replace_all(code, "__local ", "__shared__ ");
        Self::regex_replace_all(code, r"__constant\b", "__constant__");
        Self::replace_all(code, "__private ", "");
        Self::replace_all(code, "__private", "");
    }

    pub fn translate_work_item_functions(code: &mut String) {
        for (from, to) in [
            (
                "get_global_id(0)",
                "((int)(blockDim.x * blockIdx.x + threadIdx.x))",
            ),
            (
                "get_global_id(1)",
                "((int)(blockDim.y * blockIdx.y + threadIdx.y))",
            ),
            (
                "get_global_id(2)",
                "((int)(blockDim.z * blockIdx.z + threadIdx.z))",
            ),
            ("get_local_id(0)", "((int)threadIdx.x)"),
            ("get_local_id(1)", "((int)threadIdx.y)"),
            ("get_local_id(2)", "((int)threadIdx.z)"),
            ("get_group_id(0)", "((int)blockIdx.x)"),
            ("get_group_id(1)", "((int)blockIdx.y)"),
            ("get_group_id(2)", "((int)blockIdx.z)"),
            ("get_local_size(0)", "((int)blockDim.x)"),
            ("get_local_size(1)", "((int)blockDim.y)"),
            ("get_local_size(2)", "((int)blockDim.z)"),
            ("get_global_size(0)", "((int)(gridDim.x * blockDim.x))"),
            ("get_global_size(1)", "((int)(gridDim.y * blockDim.y))"),
            ("get_global_size(2)", "((int)(gridDim.z * blockDim.z))"),
            ("get_num_groups(0)", "((int)gridDim.x)"),
            ("get_num_groups(1)", "((int)gridDim.y)"),
            ("get_num_groups(2)", "((int)gridDim.z)"),
        ] {
            Self::replace_all(code, from, to);
        }
    }

    pub fn translate_synchronization(code: &mut String) {
        for (from, to) in [
            ("barrier(CLK_LOCAL_MEM_FENCE)", "__syncthreads()"),
            ("barrier(CLK_GLOBAL_MEM_FENCE)", "__syncthreads()"),
            (
                "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)",
                "__syncthreads()",
            ),
            (
                "barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)",
                "__syncthreads()",
            ),
            ("mem_fence(CLK_GLOBAL_MEM_FENCE)", "__threadfence()"),
            ("mem_fence(CLK_LOCAL_MEM_FENCE)", "__threadfence_block()"),
            ("read_mem_fence(CLK_GLOBAL_MEM_FENCE)", "__threadfence()"),
            ("write_mem_fence(CLK_GLOBAL_MEM_FENCE)", "__threadfence()"),
        ] {
            Self::replace_all(code, from, to);
        }
    }

    pub fn translate_vector_constructors(code: &mut String) {
        // Translate functional-style vector casts first: (typeN)(a, b, ...) -> make_typeN(a, b, ...)
        // Must be done before brace-style to avoid confusion
        let vector_type_names = [
            "char2", "char3", "char4", "uchar2", "uchar3", "uchar4", "short2", "short3", "short4",
            "ushort2", "ushort3", "ushort4", "int2", "int3", "int4", "uint2", "uint3", "uint4",
            "long2", "long3", "long4", "ulong2", "ulong3", "ulong4", "float2", "float3", "float4",
            "double2", "double3", "double4",
        ];

        for type_name in vector_type_names {
            let pattern = "(".to_string() + type_name + ")(";
            let replacement = "make_".to_string() + type_name + "(";
            let mut pos = 0;
            while let Some(found) = code[pos..].find(&pattern) {
                pos += found;
                // Verify this is actually a cast (preceded by space, operator, or paren, not alphanumeric)
                let should_replace = pos == 0
                    || code[..pos]
                        .chars()
                        .next_back()
                        .is_none_or(|ch| !ch.is_ascii_alphanumeric());
                if should_replace {
                    code.replace_range(pos..pos + pattern.len(), &replacement);
                    pos += replacement.len();
                } else {
                    pos += pattern.len();
                }
            }
        }

        // List of all standard OpenCL vector types and their sizes.
        // OpenCL compound literal:  (typeN){ a, b, ... }
        // CUDA equivalent:          make_typeN( a, b, ... )
        let vector_constructors = [
            // char
            ("(char2){", "make_char2("),
            ("(char3){", "make_char3("),
            ("(char4){", "make_char4("),
            // uchar
            ("(uchar2){", "make_uchar2("),
            ("(uchar3){", "make_uchar3("),
            ("(uchar4){", "make_uchar4("),
            // short
            ("(short2){", "make_short2("),
            ("(short3){", "make_short3("),
            ("(short4){", "make_short4("),
            // ushort
            ("(ushort2){", "make_ushort2("),
            ("(ushort3){", "make_ushort3("),
            ("(ushort4){", "make_ushort4("),
            // int
            ("(int2){", "make_int2("),
            ("(int3){", "make_int3("),
            ("(int4){", "make_int4("),
            // uint
            ("(uint2){", "make_uint2("),
            ("(uint3){", "make_uint3("),
            ("(uint4){", "make_uint4("),
            // long
            ("(long2){", "make_long2("),
            ("(long3){", "make_long3("),
            ("(long4){", "make_long4("),
            // ulong
            ("(ulong2){", "make_ulong2("),
            ("(ulong3){", "make_ulong3("),
            ("(ulong4){", "make_ulong4("),
            // float
            ("(float2){", "make_float2("),
            ("(float3){", "make_float3("),
            ("(float4){", "make_float4("),
            // double
            ("(double2){", "make_double2("),
            ("(double3){", "make_double3("),
            ("(double4){", "make_double4("),
        ];

        for (opencl_ctor, cuda_ctor) in vector_constructors {
            Self::replace_vector_constructor(code, &opencl_ctor, &cuda_ctor);
        }
    }

    pub fn translate_vector_access(code: &mut String) {
        for (pattern, replacement) in [
            (r"\.s0\b", ".x"),
            (r"\.s1\b", ".y"),
            (r"\.s2\b", ".z"),
            (r"\.s3\b", ".w"),
        ] {
            Self::regex_replace_all(code, pattern, replacement);
        }
    }

    pub fn translate_atomics(code: &mut String) {
        for (from, to) in [
            ("atomic_add(", "atomicAdd("),
            ("atomic_sub(", "atomicSub("),
            ("atomic_xchg(", "atomicExch("),
            ("atomic_inc(", "atomicAdd("),
            ("atomic_dec(", "atomicSub("),
            ("atomic_min(", "atomicMin("),
            ("atomic_max(", "atomicMax("),
            ("atomic_and(", "atomicAnd("),
            ("atomic_or(", "atomicOr("),
            ("atomic_xor(", "atomicXor("),
            ("atomic_cmpxchg(", "atomicCAS("),
            ("atomic_fetch_add(", "atomicAdd("),
            ("atomic_fetch_sub(", "atomicSub("),
            ("atomic_fetch_min(", "atomicMin("),
            ("atomic_fetch_max(", "atomicMax("),
            ("atomic_fetch_and(", "atomicAnd("),
            ("atomic_fetch_or(", "atomicOr("),
            ("atomic_fetch_xor(", "atomicXor("),
        ] {
            Self::replace_all(code, from, to);
        }
    }

    pub fn translate_type_conversions(code: &mut String) {
        // OpenCL convert_<type>(x) -> CUDA cast or make_ function
        //
        // Scalar: convert_int(x)     -> (int)(x)
        // Vector: convert_float4(x)  -> make_float4(...)  -- imperfect but functional for simple cases

        // Handle scalar conversions with optional rounding/saturation suffixes
        // e.g., convert_int_rte, convert_float_sat, convert_int_sat_rte
        for type_name in SCALAR_TYPES {
            // Match convert_<type> possibly followed by _sat, _rte, _rtz, _rtp, _rtn
            // We strip the rounding/saturation suffix (CUDA doesn't have direct equivalents for casts)
            let mut pattern = "convert_".to_string() + type_name + "_sat_rte(";
            let replacement = "(".to_string() + type_name + ")(";
            Self::replace_all(code, &pattern, &replacement);
            pattern = "convert_".to_string() + type_name + "_sat_rtz(";
            Self::replace_all(code, &pattern, &replacement);
            pattern = "convert_".to_string() + type_name + "_sat_rtp(";
            Self::replace_all(code, &pattern, &replacement);
            pattern = "convert_".to_string() + type_name + "_sat_rtn(";
            Self::replace_all(code, &pattern, &replacement);
            pattern = "convert_".to_string() + type_name + "_sat(";
            Self::replace_all(code, &pattern, &replacement);
            pattern = "convert_".to_string() + type_name + "_rte(";
            Self::replace_all(code, &pattern, &replacement);
            pattern = "convert_".to_string() + type_name + "_rtz(";
            Self::replace_all(code, &pattern, &replacement);
            pattern = "convert_".to_string() + type_name + "_rtp(";
            Self::replace_all(code, &pattern, &replacement);
            pattern = "convert_".to_string() + type_name + "_rtn(";
            Self::replace_all(code, &pattern, &replacement);
            pattern = "convert_".to_string() + type_name + "(";
            Self::replace_all(code, &pattern, &replacement);
        }

        // Vector conversions: convert_<type>N(...) -> make_<type>N(...)
        // This is a rough approximation -- true component-wise conversion would need per-element casts
        for type_name in SCALAR_TYPES {
            for n in ["2", "3", "4", "8", "16"] {
                let from = "convert_".to_string() + type_name + n + "(";
                let to = "make_".to_string() + type_name + n + "(";
                Self::replace_all(code, &from, &to);
            }
        }

        // as_<type> reinterpret casts -> reinterpret_cast or __<type>_as_<type> intrinsics
        // Basic approximation: as_float(x) -> __int_as_float(x), etc.
        Self::replace_all(code, "as_float(", "__int_as_float(");
        Self::replace_all(code, "as_int(", "__float_as_int(");
        Self::replace_all(code, "as_uint(", "__float_as_uint(");
    }

    pub fn translate_math_functions(code: &mut String) {
        for (from, to) in [
            ("native_sin(", "__sinf("),
            ("native_cos(", "__cosf("),
            ("native_tan(", "__tanf("),
            ("native_exp(", "__expf("),
            ("native_exp2(", "exp2f("),
            ("native_exp10(", "__exp10f("),
            ("native_log(", "__logf("),
            ("native_log2(", "__log2f("),
            ("native_log10(", "__log10f("),
            ("native_sqrt(", "__fsqrt_rn("),
            ("native_rsqrt(", "rsqrtf("),
            ("native_powr(", "__powf("),
            ("native_recip(", "__frcp_rn("),
            ("native_divide(", "__fdividef("),
            ("half_sin(", "__sinf("),
            ("half_cos(", "__cosf("),
            ("half_tan(", "__tanf("),
            ("half_exp(", "__expf("),
            ("half_exp2(", "exp2f("),
            ("half_exp10(", "__exp10f("),
            ("half_log(", "__logf("),
            ("half_log2(", "__log2f("),
            ("half_log10(", "__log10f("),
            ("half_sqrt(", "__fsqrt_rn("),
            ("half_rsqrt(", "rsqrtf("),
            ("half_powr(", "__powf("),
            ("half_recip(", "__frcp_rn("),
            ("half_divide(", "__fdividef("),
            ("mad(", "fma("),
        ] {
            Self::replace_all(code, from, to);
        }
    }

    pub fn translate_image_operations(code: &mut String) {
        // Image support is highly variable depending on the framework.
        // Below is a stub that handles the most common patterns.
        //
        // In many GPGPU frameworks (e.g., CLIJ), images are actually passed as
        // flat arrays with index macros, so the image types rarely appear raw.
        //
        // For raw OpenCL image usage, a full translation would require
        // CUDA texture objects and surface objects, which have very different APIs.

        // Image type declarations in kernel parameters:
        //   __read_only image2d_t -> cudaTextureObject_t  (approximate)
        //   __write_only image2d_t -> cudaSurfaceObject_t  (approximate)
        Self::replace_all(code, "__read_only image3d_t", "cudaTextureObject_t");
        Self::replace_all(code, "__write_only image3d_t", "cudaSurfaceObject_t");
        Self::replace_all(code, "__read_only image2d_t", "cudaTextureObject_t");
        Self::replace_all(code, "__write_only image2d_t", "cudaSurfaceObject_t");
        Self::replace_all(code, "read_only image3d_t", "cudaTextureObject_t");
        Self::replace_all(code, "write_only image3d_t", "cudaSurfaceObject_t");
        Self::replace_all(code, "read_only image2d_t", "cudaTextureObject_t");
        Self::replace_all(code, "write_only image2d_t", "cudaSurfaceObject_t");
        Self::replace_all(code, "image2d_t", "cudaTextureObject_t");
        Self::replace_all(code, "image3d_t", "cudaTextureObject_t");

        // Basic image read/write -- these are very approximate:
        // read_imagef(img, sampler, coord)  -> tex2D<float4>(img, coord.x, coord.y)
        // This would need proper parenthesis matching for a real implementation.
        // Left as markers for now:
        Self::replace_all(code, "read_imagef(", "tex2D<float4>(");
        Self::replace_all(code, "read_imagei(", "tex2D<int4>(");
        Self::replace_all(code, "read_imageui(", "tex2D<uint4>(");

        // get_image_width/height/depth
        Self::replace_all(code, "get_image_width(", "/* get_image_width */ (");
        Self::replace_all(code, "get_image_height(", "/* get_image_height */ (");
        Self::replace_all(code, "get_image_depth(", "/* get_image_depth */ (");
    }

    pub fn translate_misc_builtins(code: &mut String) {
        // select(a, b, cond) -> (cond) ? b : a
        // This is difficult to do with simple replacement because of arbitrary expressions.
        // We'll leave a comment marker for now and handle simple cases:
        // (A full implementation would need expression-aware parenthesis matching.)

        // printf -- CUDA supports printf in device code since compute capability 2.0
        // No change needed, but OpenCL printf has %v format specifiers that CUDA doesn't support.

        // CLK_* constants that might remain
        Self::replace_all(code, "CLK_LOCAL_MEM_FENCE", "0");
        Self::replace_all(code, "CLK_GLOBAL_MEM_FENCE", "0");

        // OpenCL type aliases -- use word boundaries to avoid corrupting vector types
        // (e.g. uchar2, ushort4, ulong2 are valid CUDA built-ins and must not be touched)
        Self::regex_replace_all(code, r"\buchar\b", "unsigned char");
        Self::regex_replace_all(code, r"\bushort\b", "unsigned short");
        // uint is not defined in CUDA NVRTC, so translate it
        // Must use word boundary matching to avoid replacing it in identifiers like "uint3"
        Self::regex_replace_all(code, r"\buint\b", "unsigned int");
        // "ulong" -> "unsigned long"
        Self::regex_replace_all(code, r"\bulong\b", "unsigned long");

        // Add clamp function for CUDA (not built-in)
        // OpenCL: clamp(x, lo, hi)  ->  CUDA: min(max(x, lo), hi)
        // We use a regex approach for safer matching

        // Match clamp(arg1, arg2, arg3) where arg1, arg2, arg3 are expressions
        // This regex handles simple cases; complex nested expressions may need special care.
        Self::regex_replace_all(
            code,
            r"clamp\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)",
            "min(max($1, $2), $3)",
        );
    }

    pub fn cleanup_double_qualifiers(code: &mut String) {
        for (from, to) in [
            ("__device__ __device__", "__device__"),
            ("__global__ __global__", "__global__"),
            ("__shared__ __shared__", "__shared__"),
            ("__constant__ __constant__", "__constant__"),
            ("extern \"C\" extern \"C\"", "extern \"C\""),
        ] {
            Self::replace_all(code, from, to);
        }
        Self::regex_replace_all(code, r"  +", " ");
        Self::regex_replace_all(code, r"\n{3,}", "\n\n");
    }
}

impl Drop for OpenCLToCUDATranslator {
    fn drop(&mut self) {}
}

const SCALAR_TYPES: &[&str] = &[
    "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "float", "double",
];

#[cfg(test)]
mod tests {
    use super::OpenCLToCUDATranslator;

    #[test]
    fn translates_kernel_qualifiers_and_work_items() {
        let source = "__kernel void add(__global float *out) { int x = get_global_id(0); }";

        let translated = OpenCLToCUDATranslator::new().translate(source);

        assert!(translated.contains("extern \"C\" __global__ void add( float *out)"));
        assert!(translated.contains("((int)(blockDim.x * blockIdx.x + threadIdx.x))"));
    }

    #[test]
    fn translates_vector_constructors_and_accessors() {
        let source = "float4 v = (float4){a, (b + c), d, e}; float y = v.s1;";

        let translated = OpenCLToCUDATranslator::new().translate(source);

        assert!(translated.contains("make_float4(a, (b + c), d, e)"));
        assert!(translated.contains("v.y"));
    }

    #[test]
    fn translates_misc_builtins_without_corrupting_vector_types() {
        let source = "uint a; uint3 b; uchar c; float x = clamp(v, 0.0f, 1.0f);";

        let translated = OpenCLToCUDATranslator::new().translate(source);

        assert!(translated.contains("unsigned int a"));
        assert!(translated.contains("uint3 b"));
        assert!(translated.contains("unsigned char c"));
        assert!(translated.contains("min(max(v, 0.0f), 1.0f)"));
    }
}
