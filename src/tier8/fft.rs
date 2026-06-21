use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};

fn unsupported(name: &str) -> CleError {
    CleError::Other(format!(
        "{name}: CLIc FFT core helpers are not implemented in Rust yet"
    ))
}

/// Perform a 1D, 2D, or 3D FFT.
///
/// Mirrors CLIc's `fft_func`, but currently returns an explicit unsupported
/// error because Rust does not yet have CLIc's `fft::performFFT` backend.
pub fn fft(_device: &DeviceArc, _src: &ArrayPtr, _dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    Err(unsupported("fft"))
}

/// Perform a 1D, 2D, or 3D inverse FFT.
///
/// Mirrors CLIc's `ifft_func`, but currently returns an explicit unsupported
/// error because Rust does not yet have CLIc's `fft::performIFFT` backend.
pub fn ifft(_device: &DeviceArc, _src: &ArrayPtr, _dst: &ArrayPtr) -> Result<ArrayPtr> {
    Err(unsupported("ifft"))
}

/// Perform FFT-based convolution.
///
/// Mirrors CLIc's `convolve_fft_func`, but currently returns an explicit
/// unsupported error because Rust does not yet have CLIc's FFT backend helpers.
pub fn convolve_fft(
    _device: &DeviceArc,
    _src: &ArrayPtr,
    _kernel: &ArrayPtr,
    _dst: Option<ArrayPtr>,
    _correlate: bool,
) -> Result<ArrayPtr> {
    Err(unsupported("convolve_fft"))
}

/// Perform FFT-based deconvolution.
///
/// Mirrors CLIc's `deconvolve_fft_func`, but currently returns an explicit
/// unsupported error because Rust does not yet have CLIc's FFT backend helpers.
pub fn deconvolve_fft(
    _device: &DeviceArc,
    _src: &ArrayPtr,
    _psf: &ArrayPtr,
    _normalization: Option<&ArrayPtr>,
    _dst: Option<ArrayPtr>,
    _iteration: i32,
    _regularization: f32,
) -> Result<ArrayPtr> {
    Err(unsupported("deconvolve_fft"))
}
