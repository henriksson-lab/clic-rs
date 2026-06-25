//! Core FFT helper facade.
//!
//! This mirrors the public helper surface from CLIc's `clic/include/fft.hpp`.
//! The runtime path uses vendored VkFFT through a small OpenCL C shim while
//! preserving CLIc's allocation/configuration/cache shape.

use crate::array::{Array, ArrayPtr};
use crate::backend::GpuMemory;
use crate::cache::DiskCache;
use crate::device::{DeviceArc, OpenCLDevice};
use crate::error::{CleError, Result};
use crate::execution::{native_execute, ParameterValue};
use crate::types::DType;
use opencl3::memory::ClMem;
use opencl3::types::{cl_command_queue, cl_context, cl_device_id, cl_mem};
use std::any::Any;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;

const FFT_SRC: &str = include_str!("../kernels/fft.cl");

#[repr(C)]
struct ClicVkfftPlan {
    _private: [u8; 0],
}

#[derive(Default)]
pub struct FftConfiguration {
    pub number_batches: u64,
    pub size: [usize; 3],
    pub fft_dim: usize,
    pub normalize: bool,
    pub perform_r2c: bool,
    pub perform_dct: bool,
    pub is_input_formatted: bool,
    pub inverse_return_to_input_buffer: bool,
    pub input_buffer_stride: [usize; 3],
    pub load_application_from_string: bool,
    pub save_application_to_string: bool,
    pub load_application_string: Vec<u8>,
}

pub struct FftApplication {
    pub save_application_string: Vec<u8>,
}

extern "C" {
    fn clic_vkfft_perform_fft(
        device: cl_device_id,
        context: cl_context,
        queue: cl_command_queue,
        input_mem: cl_mem,
        input_size: u64,
        output_mem: cl_mem,
        output_size: u64,
        width: u64,
        height: u64,
        depth: u64,
        load_application_string: *const u8,
        load_application_string_len: u64,
        save_application_to_string: c_int,
        saved_application_string: *mut *mut u8,
        saved_application_string_len: *mut u64,
    ) -> c_int;

    fn clic_vkfft_perform_ifft(
        device: cl_device_id,
        context: cl_context,
        queue: cl_command_queue,
        input_mem: cl_mem,
        input_size: u64,
        output_mem: cl_mem,
        output_size: u64,
        width: u64,
        height: u64,
        depth: u64,
        load_application_string: *const u8,
        load_application_string_len: u64,
        save_application_to_string: c_int,
        saved_application_string: *mut *mut u8,
        saved_application_string_len: *mut u64,
    ) -> c_int;

    fn clic_vkfft_error_string(result: c_int) -> *const c_char;
    fn clic_vkfft_free(ptr: *mut c_void);

    fn clic_vkfft_create_fft_plan(
        device: cl_device_id,
        context: cl_context,
        queue: cl_command_queue,
        input_mem: cl_mem,
        input_size: u64,
        output_mem: cl_mem,
        output_size: u64,
        width: u64,
        height: u64,
        depth: u64,
        plan_out: *mut *mut ClicVkfftPlan,
    ) -> c_int;

    fn clic_vkfft_create_ifft_plan(
        device: cl_device_id,
        context: cl_context,
        queue: cl_command_queue,
        input_mem: cl_mem,
        input_size: u64,
        output_mem: cl_mem,
        output_size: u64,
        width: u64,
        height: u64,
        depth: u64,
        plan_out: *mut *mut ClicVkfftPlan,
    ) -> c_int;

    fn clic_vkfft_append_fft(plan: *mut ClicVkfftPlan, input_mem: cl_mem) -> c_int;
    fn clic_vkfft_append_ifft(plan: *mut ClicVkfftPlan) -> c_int;
    fn clic_vkfft_delete_plan(plan: *mut ClicVkfftPlan);
}

struct VkfftPlan {
    ptr: *mut ClicVkfftPlan,
}

// C++ calls deleteVkFFT manually. Drop keeps the same lifetime while ensuring
// FFI plans are released on every Rust early-return path.
impl Drop for VkfftPlan {
    fn drop(&mut self) {
        unsafe {
            clic_vkfft_delete_plan(self.ptr);
        }
    }
}

/// Get the padded shape needed to avoid circular convolution artifacts.
pub fn fft_pad_shape(image_shape: [usize; 3], kernel_shape: [usize; 3]) -> [usize; 3] {
    [
        image_shape[0] + 2 * (kernel_shape[0] / 2),
        image_shape[1] + 2 * (kernel_shape[1] / 2),
        image_shape[2] + 2 * (kernel_shape[2] / 2),
    ]
}

/// Create a hermitian complex buffer matching CLIc's FFT output layout.
///
/// CLIc stores complex values interleaved in a buffer shaped as
/// `[(width / 2 + 1) * 2, height, depth]` with `DType::Complex`.
pub fn create_hermitian(input: &ArrayPtr) -> Result<ArrayPtr> {
    let input = input.lock().unwrap();
    let hermitian_width = input.width() / 2 + 1;

    Array::create(
        hermitian_width * 2,
        input.height(),
        input.depth(),
        input.dim(),
        DType::Complex,
        input.mtype(),
        input.device(),
    )
}

pub fn configure(array: &ArrayPtr, configuration: &mut FftConfiguration) {
    let array = array.lock().unwrap();
    configuration.number_batches = 1;
    configuration.size[0] = array.width();
    configuration.size[1] = array.height();
    configuration.size[2] = array.depth();
    configuration.fft_dim = 1;
    if configuration.size[1] > 1 {
        configuration.fft_dim += 1;
    }
    if configuration.size[2] > 1 {
        configuration.fft_dim += 1;
    }

    configuration.normalize = true;
    configuration.perform_r2c = true;
    configuration.perform_dct = false;
    configuration.is_input_formatted = true;
    configuration.inverse_return_to_input_buffer = true;
    configuration.input_buffer_stride[0] = configuration.size[0];
    configuration.input_buffer_stride[1] =
        configuration.input_buffer_stride[0] * configuration.size[1];
    configuration.input_buffer_stride[2] =
        configuration.input_buffer_stride[1] * configuration.size[2];
}

pub fn get_cache_path(output: &ArrayPtr, device: &DeviceArc) -> PathBuf {
    let output = output.lock().unwrap();
    let source_hash = DiskCache::hash(&format!(
        "{},{},{},{}",
        output.width(),
        output.height(),
        output.depth(),
        output.dim()
    ));
    DiskCache::instance().get_file_path(&device.device_hash(), &source_hash, "bin")
}

pub fn load_kernel_cache(binary_path: &Path, configuration: &mut FftConfiguration) -> Result<bool> {
    match std::fs::read(binary_path) {
        Ok(bytes) => {
            configuration.load_application_string = bytes;
            configuration.load_application_from_string = true;
            configuration.save_application_to_string = false;
            Ok(true)
        }
        Err(_) => Ok(false),
    }
}

pub fn save_kernel_cache(binary_path: &Path, app: &FftApplication) -> Result<()> {
    if let Some(parent) = binary_path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            CleError::Other(format!(
                "save_kernel_cache: failed to create {}: {error}",
                parent.display()
            ))
        })?;
    }
    std::fs::write(binary_path, &app.save_application_string).map_err(|error| {
        CleError::Other(format!(
            "save_kernel_cache: failed to write {}: {error}",
            binary_path.display()
        ))
    })
}

macro_rules! vkfft_error {
    ($prefix:expr, $code:expr) => {{
        let code = $code;
        let message = unsafe {
            let ptr = clic_vkfft_error_string(code);
            if ptr.is_null() {
                format!("unknown VkFFT error {code}")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        CleError::Other(format!("{}: {}", $prefix, message))
    }};
}

/// Execute an FFT pointwise operation kernel.
///
/// Mirrors CLIc's `execOperationKernel`.
pub fn exec_operation_kernel(
    device: &DeviceArc,
    name: &str,
    input_a: &ArrayPtr,
    input_b: &ArrayPtr,
    output: &ArrayPtr,
    n_elements: u32,
) -> Result<ArrayPtr> {
    let local_item_size = 256.min(device.get_maximum_work_group_size());
    let global_item_size =
        ((n_elements as f64 / local_item_size as f64).ceil() as usize) * local_item_size;
    let params = vec![
        ("a", ParameterValue::Array(input_a.clone())),
        ("b", ParameterValue::Array(input_b.clone())),
        ("c", ParameterValue::Array(output.clone())),
        ("n", ParameterValue::Uint(n_elements)),
    ];
    native_execute(
        device,
        (name, FFT_SRC),
        &params,
        [global_item_size, 1, 1],
        [local_item_size, 1, 1],
    )?;
    Ok(output.clone())
}

/// Execute the in-place remove-small-values FFT helper kernel.
pub fn exec_remove_small_values(
    device: &DeviceArc,
    buffer: &ArrayPtr,
    n_elements: u32,
) -> Result<()> {
    let local_item_size = 256.min(device.get_maximum_work_group_size());
    let global_item_size =
        ((n_elements as f64 / local_item_size as f64).ceil() as usize) * local_item_size;
    let params = vec![
        ("a", ParameterValue::Array(buffer.clone())),
        ("n", ParameterValue::Uint(n_elements)),
    ];
    native_execute(
        device,
        ("removeSmallValues", FFT_SRC),
        &params,
        [global_item_size, 1, 1],
        [local_item_size, 1, 1],
    )
}

/// Execute the total variation term FFT helper kernel.
pub fn exec_total_variation_term(
    device: &DeviceArc,
    estimate: &ArrayPtr,
    correction: &ArrayPtr,
    variation: &ArrayPtr,
    hx: f32,
    hy: f32,
    hz: f32,
    regularization_factor: f32,
) -> Result<()> {
    let (nx, ny, nz) = {
        let estimate = estimate.lock().unwrap();
        (
            estimate.width() as u32,
            estimate.height() as u32,
            estimate.depth() as u32,
        )
    };
    let params = vec![
        ("estimate", ParameterValue::Array(estimate.clone())),
        ("correction", ParameterValue::Array(correction.clone())),
        ("variation", ParameterValue::Array(variation.clone())),
        ("Nx", ParameterValue::Uint(nx)),
        ("Ny", ParameterValue::Uint(ny)),
        ("Nz", ParameterValue::Uint(nz)),
        ("hx", ParameterValue::Float(hx)),
        ("hy", ParameterValue::Float(hy)),
        ("hz", ParameterValue::Float(hz)),
        (
            "regularizationFactor",
            ParameterValue::Float(regularization_factor),
        ),
    ];
    native_execute(
        device,
        ("totalVariationTerm", FFT_SRC),
        &params,
        [nx as usize, ny as usize, nz as usize],
        [0, 0, 0],
    )
}

/// Perform a forward FFT.
pub fn perform_fft(input: &ArrayPtr, output: Option<ArrayPtr>) -> Result<ArrayPtr> {
    // create hermitian buffer if output is not provided
    let output = match output {
        Some(output) => output,
        None => create_hermitian(input)?,
    };

    let (width, height, depth) = {
        let input = input.lock().unwrap();
        (input.width(), input.height(), input.depth())
    };

    // configure VkFFT
    let mut configuration = FftConfiguration::default();
    configure(input, &mut configuration);

    // manage jit-cache system
    let device = input.lock().unwrap().device().clone();
    let use_cache = DiskCache::instance().is_enabled();
    let mut binary_path = PathBuf::new();
    if use_cache {
        binary_path = get_cache_path(input, &device);
        if !load_kernel_cache(&binary_path, &mut configuration)? {
            configuration.load_application_from_string = false;
            configuration.save_application_to_string = true;
        }
    }

    let input_mem = {
        let mem = input
            .lock()
            .unwrap()
            .get_ptr()
            .ok_or(CleError::NotAllocated)?;
        match mem.as_ref() {
            GpuMemory::Buffer(buffer) => buffer.lock().unwrap().get(),
            GpuMemory::Image(image) => image.lock().unwrap().get(),
        }
    };
    let output_mem = {
        let mem = output
            .lock()
            .unwrap()
            .get_ptr()
            .ok_or(CleError::NotAllocated)?;
        match mem.as_ref() {
            GpuMemory::Buffer(buffer) => buffer.lock().unwrap().get(),
            GpuMemory::Image(image) => image.lock().unwrap().get(),
        }
    };
    let psize = output.lock().unwrap().bitsize() as u64;
    let psizein = input.lock().unwrap().bitsize() as u64;
    let ocl = (device.as_ref() as &dyn Any)
        .downcast_ref::<OpenCLDevice>()
        .ok_or_else(|| {
            CleError::Other("FFT VkFFT runtime requires an OpenCL device".to_string())
        })?;
    let (load_ptr, load_len) = if configuration.load_application_from_string {
        (
            configuration.load_application_string.as_ptr(),
            configuration.load_application_string.len() as u64,
        )
    } else {
        (ptr::null(), 0)
    };
    let mut saved_ptr = ptr::null_mut();
    let mut saved_len = 0_u64;
    let res = unsafe {
        clic_vkfft_perform_fft(
            ocl.get_cl_device(),
            ocl.get_cl_context(),
            ocl.get_cl_command_queue(),
            input_mem,
            psizein,
            output_mem,
            psize,
            width as u64,
            height as u64,
            depth as u64,
            load_ptr,
            load_len,
            i32::from(use_cache && configuration.save_application_to_string),
            &mut saved_ptr,
            &mut saved_len,
        )
    };
    if res != 0 {
        if !saved_ptr.is_null() {
            unsafe {
                clic_vkfft_free(saved_ptr.cast::<c_void>());
            }
        }
        return Err(vkfft_error!("perform_fft", res));
    }
    if use_cache && configuration.save_application_to_string && !saved_ptr.is_null() {
        let result = if saved_len == 0 {
            Ok(())
        } else {
            let bytes = unsafe { slice::from_raw_parts(saved_ptr, saved_len as usize) };
            let app = FftApplication {
                save_application_string: bytes.to_vec(),
            };
            save_kernel_cache(&binary_path, &app)
        };
        unsafe {
            clic_vkfft_free(saved_ptr.cast::<c_void>());
        }
        result?;
    }
    device.finish();

    Ok(output)
}

/// Perform an inverse FFT.
pub fn perform_ifft(input: &ArrayPtr, output: &ArrayPtr) -> Result<()> {
    let (width, height, depth) = {
        let output = output.lock().unwrap();
        (output.width(), output.height(), output.depth())
    };

    // configure VkFFT
    let mut configuration = FftConfiguration::default();
    configure(output, &mut configuration);

    // manage jit-cache system
    let device = input.lock().unwrap().device().clone();
    let use_cache = DiskCache::instance().is_enabled();
    let mut binary_path = PathBuf::new();
    if use_cache {
        binary_path = get_cache_path(output, &device);
        if !load_kernel_cache(&binary_path, &mut configuration)? {
            configuration.load_application_from_string = false;
            configuration.save_application_to_string = true;
        }
    }

    let input_mem = {
        let mem = input
            .lock()
            .unwrap()
            .get_ptr()
            .ok_or(CleError::NotAllocated)?;
        match mem.as_ref() {
            GpuMemory::Buffer(buffer) => buffer.lock().unwrap().get(),
            GpuMemory::Image(image) => image.lock().unwrap().get(),
        }
    };
    let output_mem = {
        let mem = output
            .lock()
            .unwrap()
            .get_ptr()
            .ok_or(CleError::NotAllocated)?;
        match mem.as_ref() {
            GpuMemory::Buffer(buffer) => buffer.lock().unwrap().get(),
            GpuMemory::Image(image) => image.lock().unwrap().get(),
        }
    };
    let input_size = input.lock().unwrap().bitsize() as u64;
    let output_size = output.lock().unwrap().bitsize() as u64;
    let ocl = (device.as_ref() as &dyn Any)
        .downcast_ref::<OpenCLDevice>()
        .ok_or_else(|| {
            CleError::Other("FFT VkFFT runtime requires an OpenCL device".to_string())
        })?;
    let (load_ptr, load_len) = if configuration.load_application_from_string {
        (
            configuration.load_application_string.as_ptr(),
            configuration.load_application_string.len() as u64,
        )
    } else {
        (ptr::null(), 0)
    };
    let mut saved_ptr = ptr::null_mut();
    let mut saved_len = 0_u64;
    let res = unsafe {
        clic_vkfft_perform_ifft(
            ocl.get_cl_device(),
            ocl.get_cl_context(),
            ocl.get_cl_command_queue(),
            input_mem,
            input_size,
            output_mem,
            output_size,
            width as u64,
            height as u64,
            depth as u64,
            load_ptr,
            load_len,
            i32::from(use_cache && configuration.save_application_to_string),
            &mut saved_ptr,
            &mut saved_len,
        )
    };
    if res != 0 {
        if !saved_ptr.is_null() {
            unsafe {
                clic_vkfft_free(saved_ptr.cast::<c_void>());
            }
        }
        return Err(vkfft_error!("perform_ifft", res));
    }
    if use_cache && configuration.save_application_to_string && !saved_ptr.is_null() {
        let result = if saved_len == 0 {
            Ok(())
        } else {
            let bytes = unsafe { slice::from_raw_parts(saved_ptr, saved_len as usize) };
            let app = FftApplication {
                save_application_string: bytes.to_vec(),
            };
            save_kernel_cache(&binary_path, &app)
        };
        unsafe {
            clic_vkfft_free(saved_ptr.cast::<c_void>());
        }
        result?;
    }
    device.finish();

    Ok(())
}

/// Perform FFT-based convolution.
pub fn perform_convolution(
    input: &ArrayPtr,
    psf: &ArrayPtr,
    output: &ArrayPtr,
    correlate: bool,
) -> Result<()> {
    let device = input.lock().unwrap().device().clone();
    // forward fft of input and psf
    let fft_input = perform_fft(input, None)?;
    let fft_psf = perform_fft(psf, None)?;
    let fft_out = Array::create_from_array(&fft_psf)?;

    // complex multiply input and psf
    let kernel_name = if correlate {
        "vecComplexConjugateMultiply"
    } else {
        "vecComplexMultiply"
    };
    let n_elements = (fft_input.lock().unwrap().size() / 2) as u32;
    exec_operation_kernel(
        &device,
        kernel_name,
        &fft_input,
        &fft_psf,
        &fft_out,
        n_elements,
    )?;

    // Inverse to get convolved
    perform_ifft(&fft_out, output)
}

/// Perform Richardson-Lucy deconvolution using FFT helpers.
pub fn perform_deconvolution(
    observe: &ArrayPtr,
    psf: &ArrayPtr,
    normal: Option<&ArrayPtr>,
    estimate: Option<ArrayPtr>,
    iterations: usize,
    regularization: f32,
) -> Result<ArrayPtr> {
    let device = observe.lock().unwrap().device().clone();
    let use_tv = regularization > 0.0;

    let reblurred = Array::create_from_array(observe)?;
    let estimate = match estimate {
        Some(estimate) => estimate,
        None => {
            let estimate = Array::create_from_array(observe)?;
            observe.lock().unwrap().copy_to(&estimate)?;
            estimate
        }
    };
    let fft_psf = create_hermitian(psf)?;
    let fft_estimate = create_hermitian(&estimate)?;

    let variation = if use_tv {
        Some(Array::create_from_array(observe)?)
    } else {
        None
    };

    if let Some(normal) = normal {
        let n_elements = normal.lock().unwrap().size() as u32;
        exec_remove_small_values(&device, normal, n_elements)?;
    }

    // FFT of PSF (single init+exec+delete - done only once)
    perform_fft(psf, Some(fft_psf.clone()))?;

    let real_size = estimate.lock().unwrap().bitsize() as u64;
    let complex_size = fft_estimate.lock().unwrap().bitsize() as u64;
    let (width, height, depth) = {
        let estimate = estimate.lock().unwrap();
        (estimate.width(), estimate.height(), estimate.depth())
    };
    let ocl = (device.as_ref() as &dyn Any)
        .downcast_ref::<OpenCLDevice>()
        .ok_or_else(|| {
            CleError::Other("FFT VkFFT runtime requires an OpenCL device".to_string())
        })?;
    let fft_real_in_mem = {
        let mem = estimate
            .lock()
            .unwrap()
            .get_ptr()
            .ok_or(CleError::NotAllocated)?;
        match mem.as_ref() {
            GpuMemory::Buffer(buffer) => buffer.lock().unwrap().get(),
            GpuMemory::Image(image) => image.lock().unwrap().get(),
        }
    };
    let fft_complex_mem = {
        let mem = fft_estimate
            .lock()
            .unwrap()
            .get_ptr()
            .ok_or(CleError::NotAllocated)?;
        match mem.as_ref() {
            GpuMemory::Buffer(buffer) => buffer.lock().unwrap().get(),
            GpuMemory::Image(image) => image.lock().unwrap().get(),
        }
    };
    let ifft_real_out = {
        let mem = reblurred
            .lock()
            .unwrap()
            .get_ptr()
            .ok_or(CleError::NotAllocated)?;
        match mem.as_ref() {
            GpuMemory::Buffer(buffer) => buffer.lock().unwrap().get(),
            GpuMemory::Image(image) => image.lock().unwrap().get(),
        }
    };

    let mut fft_plan_ptr = ptr::null_mut();
    let res = unsafe {
        clic_vkfft_create_fft_plan(
            ocl.get_cl_device(),
            ocl.get_cl_context(),
            ocl.get_cl_command_queue(),
            fft_real_in_mem,
            real_size,
            fft_complex_mem,
            complex_size,
            width as u64,
            height as u64,
            depth as u64,
            &mut fft_plan_ptr,
        )
    };
    if res != 0 {
        return Err(vkfft_error!("perform_deconvolution forward init", res));
    }
    let fft_plan = VkfftPlan { ptr: fft_plan_ptr };

    let mut ifft_plan_ptr = ptr::null_mut();
    let res = unsafe {
        clic_vkfft_create_ifft_plan(
            ocl.get_cl_device(),
            ocl.get_cl_context(),
            ocl.get_cl_command_queue(),
            fft_complex_mem,
            complex_size,
            ifft_real_out,
            real_size,
            width as u64,
            height as u64,
            depth as u64,
            &mut ifft_plan_ptr,
        )
    };
    if res != 0 {
        return Err(vkfft_error!("perform_deconvolution inverse init", res));
    }
    let ifft_plan = VkfftPlan { ptr: ifft_plan_ptr };

    for _ in 0..iterations {
        // Forward FFT of estimate -> fft_estimate
        let fft_real_in_mem = {
            let mem = estimate
                .lock()
                .unwrap()
                .get_ptr()
                .ok_or(CleError::NotAllocated)?;
            match mem.as_ref() {
                GpuMemory::Buffer(buffer) => buffer.lock().unwrap().get(),
                GpuMemory::Image(image) => image.lock().unwrap().get(),
            }
        };
        let res = unsafe { clic_vkfft_append_fft(fft_plan.ptr, fft_real_in_mem) };
        if res != 0 {
            return Err(vkfft_error!("perform_deconvolution forward append", res));
        }

        // complex multiply: fft_estimate x fft_psf -> fft_estimate
        let n_elements = (fft_estimate.lock().unwrap().size() / 2) as u32;
        exec_operation_kernel(
            &device,
            "vecComplexMultiply",
            &fft_estimate,
            &fft_psf,
            &fft_estimate,
            n_elements,
        )?;

        // Inverse FFT: fft_estimate -> reblurred
        let res = unsafe { clic_vkfft_append_ifft(ifft_plan.ptr) };
        if res != 0 {
            return Err(vkfft_error!("perform_deconvolution inverse append", res));
        }

        // Divide observed by reblurred
        let n_elements = observe.lock().unwrap().size() as u32;
        exec_operation_kernel(
            &device, "vecDiv", observe, &reblurred, &reblurred, n_elements,
        )?;

        // Forward FFT of reblurred -> fft_estimate
        let fft_real_in_mem = {
            let mem = reblurred
                .lock()
                .unwrap()
                .get_ptr()
                .ok_or(CleError::NotAllocated)?;
            match mem.as_ref() {
                GpuMemory::Buffer(buffer) => buffer.lock().unwrap().get(),
                GpuMemory::Image(image) => image.lock().unwrap().get(),
            }
        };
        let res = unsafe { clic_vkfft_append_fft(fft_plan.ptr, fft_real_in_mem) };
        if res != 0 {
            return Err(vkfft_error!("perform_deconvolution forward append", res));
        }

        // Correlate: complex conjugate multiply of fft_estimate with fft_psf
        let n_elements = (fft_estimate.lock().unwrap().size() / 2) as u32;
        exec_operation_kernel(
            &device,
            "vecComplexConjugateMultiply",
            &fft_estimate,
            &fft_psf,
            &fft_estimate,
            n_elements,
        )?;

        // Inverse FFT: fft_estimate -> reblurred
        let res = unsafe { clic_vkfft_append_ifft(ifft_plan.ptr) };
        if res != 0 {
            return Err(vkfft_error!("perform_deconvolution inverse append", res));
        }

        if let Some(variation) = &variation {
            exec_total_variation_term(
                &device,
                &estimate,
                &reblurred,
                variation,
                1.0,
                1.0,
                3.0,
                regularization,
            )?;
            let n_elements = estimate.lock().unwrap().size() as u32;
            exec_operation_kernel(
                &device, "vecMul", &estimate, variation, &estimate, n_elements,
            )?;
        } else {
            let n_elements = estimate.lock().unwrap().size() as u32;
            exec_operation_kernel(
                &device, "vecMul", &estimate, &reblurred, &estimate, n_elements,
            )?;
        }

        if let Some(normal) = normal {
            let n_elements = estimate.lock().unwrap().size() as u32;
            exec_operation_kernel(&device, "vecDiv", &estimate, normal, &estimate, n_elements)?;
        }

        device.finish();
    }

    Ok(estimate)
}
