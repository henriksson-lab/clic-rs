//! Core FFT helper facade.
//!
//! This mirrors the public helper surface from CLIc's `clic/include/fft.hpp`.
//! Rust does not currently have a VkFFT/OpenCL/CUDA FFT backend, so backend
//! entry points return explicit unsupported errors.

use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::types::DType;
use crate::utils::fft_smooth_shape as utils_fft_smooth_shape;

fn unsupported(name: &str) -> CleError {
    CleError::Other(format!(
        "{name}: CLIc FFT backend helpers are not implemented in Rust yet"
    ))
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

/// Get an FFT-friendly shape by rounding each dimension up to a smooth number.
pub fn fft_smooth_shape(shape: [usize; 3]) -> [usize; 3] {
    utils_fft_smooth_shape(shape)
}

/// Get the padded shape needed to avoid circular convolution artifacts.
pub fn fft_pad_shape(image_shape: [usize; 3], kernel_shape: [usize; 3]) -> [usize; 3] {
    [
        image_shape[0] + 2 * (kernel_shape[0] / 2),
        image_shape[1] + 2 * (kernel_shape[1] / 2),
        image_shape[2] + 2 * (kernel_shape[2] / 2),
    ]
}

/// Execute an FFT pointwise operation kernel.
///
/// Mirrors CLIc's `execOperationKernel`, but no Rust FFT kernel backend is
/// currently available.
pub fn exec_operation_kernel(
    _device: &DeviceArc,
    _name: &str,
    _buffer_a: &ArrayPtr,
    _buffer_b: &ArrayPtr,
    _buffer_out: &ArrayPtr,
    _n_elements: u32,
) -> Result<ArrayPtr> {
    Err(unsupported("exec_operation_kernel"))
}

/// Execute the in-place remove-small-values FFT helper kernel.
pub fn exec_remove_small_values(
    _device: &DeviceArc,
    _buffer: &ArrayPtr,
    _n_elements: u32,
) -> Result<()> {
    Err(unsupported("exec_remove_small_values"))
}

/// Execute the total variation term FFT helper kernel.
pub fn exec_total_variation_term(
    _device: &DeviceArc,
    _estimate: &ArrayPtr,
    _correction: &ArrayPtr,
    _variation: &ArrayPtr,
    _hx: f32,
    _hy: f32,
    _hz: f32,
    _regularization_factor: f32,
) -> Result<()> {
    Err(unsupported("exec_total_variation_term"))
}

/// Perform a forward FFT.
pub fn perform_fft(_input: &ArrayPtr, _output: Option<ArrayPtr>) -> Result<ArrayPtr> {
    Err(unsupported("perform_fft"))
}

/// Perform an inverse FFT.
pub fn perform_ifft(_input: &ArrayPtr, _output: &ArrayPtr) -> Result<()> {
    Err(unsupported("perform_ifft"))
}

/// Perform FFT-based convolution.
pub fn perform_convolution(
    _input: &ArrayPtr,
    _psf: &ArrayPtr,
    _output: &ArrayPtr,
    _correlate: bool,
) -> Result<()> {
    Err(unsupported("perform_convolution"))
}

/// Perform Richardson-Lucy deconvolution using FFT helpers.
pub fn perform_deconvolution(
    _observe: &ArrayPtr,
    _psf: &ArrayPtr,
    _normal: Option<&ArrayPtr>,
    _estimate: Option<ArrayPtr>,
    _iterations: usize,
    _regularization: f32,
) -> Result<ArrayPtr> {
    Err(unsupported("perform_deconvolution"))
}
