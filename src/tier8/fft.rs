use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::types::DType;
use crate::{fft as fft_core, tier0, tier1};

/// Perform a 1D, 2D, or 3D FFT.
pub fn fft(_device: &DeviceArc, src: &ArrayPtr, dst: Option<ArrayPtr>) -> Result<ArrayPtr> {
    fft_core::perform_fft(src, dst)
}

/// Perform a 1D, 2D, or 3D inverse FFT.
pub fn ifft(_device: &DeviceArc, src: &ArrayPtr, dst: &ArrayPtr) -> Result<ArrayPtr> {
    if src.lock().unwrap().dtype() != DType::Complex {
        eprintln!(
            "Warning: ifft input buffer expect COMPLEX type, but got {:?}.",
            src.lock().unwrap().dtype()
        );
    }

    fft_core::perform_ifft(src, dst)?;

    Ok(dst.clone())
}

/// Perform FFT-based convolution.
pub fn convolve_fft(
    device: &DeviceArc,
    src: &ArrayPtr,
    kernel: &ArrayPtr,
    dst: Option<ArrayPtr>,
    correlate: bool,
) -> Result<ArrayPtr> {
    let image_shape = {
        let src = src.lock().unwrap();
        [src.width(), src.height(), src.depth()]
    };
    let kernel_shape = {
        let kernel = kernel.lock().unwrap();
        [kernel.width(), kernel.height(), kernel.depth()]
    };
    if kernel_shape[0] > image_shape[0]
        || kernel_shape[1] > image_shape[1]
        || kernel_shape[2] > image_shape[2]
    {
        return Err(CleError::Other(format!(
            "Error: Kernel size is larger than the input buffer size. Kernel size: {}x{}x{} Input size: {}x{}x{}",
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            image_shape[0],
            image_shape[1],
            image_shape[2]
        )));
    }

    // get the next smooth shape for the input and kernel to facilitate the fft
    // let pad_shape = fft_core::fft_pad_shape(image_shape, kernel_shape);
    let pad_shape = [image_shape[0], image_shape[1], image_shape[2]];
    let smoothed_shape = crate::utils::fft_smooth_shape(pad_shape);

    // check if smooth size differs from the input size, if yes pad input and save the padding size for unpadding
    let mut padded = false;
    let mut pad_input = src.clone();
    if smoothed_shape[0] != image_shape[0]
        || smoothed_shape[1] != image_shape[1]
        || smoothed_shape[2] != image_shape[2]
    {
        pad_input = tier1::pad(
            device,
            src,
            None,
            smoothed_shape[0],
            smoothed_shape[1],
            smoothed_shape[2],
            0.0,
            true,
        )?;
        padded = true;
    }

    // check if smooth size differs from the kernel size, if yes, pad the kernel
    let mut pad_kernel = kernel.clone();
    if smoothed_shape[0] != kernel_shape[0]
        || smoothed_shape[1] != kernel_shape[1]
        || smoothed_shape[2] != kernel_shape[2]
    {
        pad_kernel = tier1::pad(
            device,
            kernel,
            None,
            smoothed_shape[0],
            smoothed_shape[1],
            smoothed_shape[2],
            0.0,
            true,
        )?;
    }

    // check dst size and pad if needed, otherwise create a new buffer
    // negative shift kernel to center it at (0, 0, 0)
    let (x_center, y_center, z_center) = {
        let pad_kernel = pad_kernel.lock().unwrap();
        (
            (pad_kernel.width() as f64 / 2.0).ceil() as i32 - 1,
            (pad_kernel.height() as f64 / 2.0).ceil() as i32 - 1,
            (pad_kernel.depth() as f64 / 2.0).ceil() as i32 - 1,
        )
    };
    pad_kernel = tier1::circular_shift(device, &pad_kernel, None, -x_center, -y_center, -z_center)?;

    // perform convolution
    let dst = tier0::create_like(&pad_input, dst, DType::Unknown, device)?;
    fft_core::perform_convolution(&pad_input, &pad_kernel, &dst, correlate)?;

    // unpad the result if needed
    let dst = if padded {
        tier1::unpad(
            device,
            &dst,
            None,
            image_shape[0],
            image_shape[1],
            image_shape[2],
            true,
        )?
    } else {
        dst
    };

    Ok(dst)
}

/// Perform FFT-based deconvolution.
pub fn deconvolve_fft(
    device: &DeviceArc,
    src: &ArrayPtr,
    psf: &ArrayPtr,
    normalization: Option<&ArrayPtr>,
    dst: Option<ArrayPtr>,
    iteration: i32,
    regularization: f32,
) -> Result<ArrayPtr> {
    let image_shape = {
        let src = src.lock().unwrap();
        [src.width(), src.height(), src.depth()]
    };
    let psf_shape = {
        let psf = psf.lock().unwrap();
        [psf.width(), psf.height(), psf.depth()]
    };
    if psf_shape[0] > image_shape[0]
        || psf_shape[1] > image_shape[1]
        || psf_shape[2] > image_shape[2]
    {
        return Err(CleError::Other(format!(
            "Error: Kernel size is larger than the input buffer size. Kernel size: {}x{}x{} Input size: {}x{}x{}",
            psf_shape[0],
            psf_shape[1],
            psf_shape[2],
            image_shape[0],
            image_shape[1],
            image_shape[2]
        )));
    }

    // let pad_shape = fft_core::fft_pad_shape(image_shape, psf_shape);
    let pad_shape = [image_shape[0], image_shape[1], image_shape[2]];
    let smoothed_shape = crate::utils::fft_smooth_shape(pad_shape);

    // check if smooth size differs from the input size, if yes pad input and save the padding size for unpadding
    let mut padded = false;
    let mut pad_input = src.clone();
    if smoothed_shape[0] != image_shape[0]
        || smoothed_shape[1] != image_shape[1]
        || smoothed_shape[2] != image_shape[2]
    {
        pad_input = tier1::pad(
            device,
            src,
            None,
            smoothed_shape[0],
            smoothed_shape[1],
            smoothed_shape[2],
            0.0,
            true,
        )?;
        padded = true;
    }

    // check if smooth size differs from the kernel size, if yes, pad the kernel
    let mut pad_psf = psf.clone();
    if smoothed_shape[0] != psf_shape[0]
        || smoothed_shape[1] != psf_shape[1]
        || smoothed_shape[2] != psf_shape[2]
    {
        pad_psf = tier1::pad(
            device,
            psf,
            None,
            smoothed_shape[0],
            smoothed_shape[1],
            smoothed_shape[2],
            0.0,
            true,
        )?;
    }

    // check if smooth size differs from the kernel size, if yes, pad the kernel
    let mut pad_norm = normalization.cloned();
    if pad_norm.is_some() {
        let normalization = pad_norm.as_ref().unwrap();
        let norm_shape = {
            let normalization = normalization.lock().unwrap();
            [
                normalization.width(),
                normalization.height(),
                normalization.depth(),
            ]
        };
        if smoothed_shape[0] != norm_shape[0]
            || smoothed_shape[1] != norm_shape[1]
            || smoothed_shape[2] != norm_shape[2]
        {
            pad_norm = Some(tier1::pad(
                device,
                normalization,
                None,
                smoothed_shape[0],
                smoothed_shape[1],
                smoothed_shape[2],
                0.0,
                true,
            )?);
        }
    }

    // check dst size and pad if needed, otherwise create a new buffer
    let mut pad_dst = dst;
    if let Some(dst) = &pad_dst {
        let dst_shape = {
            let dst = dst.lock().unwrap();
            [dst.width(), dst.height(), dst.depth()]
        };
        if smoothed_shape[0] != dst_shape[0]
            || smoothed_shape[1] != dst_shape[1]
            || smoothed_shape[2] != dst_shape[2]
        {
            pad_dst = Some(tier1::pad(
                device,
                dst,
                None,
                smoothed_shape[0],
                smoothed_shape[1],
                smoothed_shape[2],
                0.0,
                true,
            )?);
        }
    } else {
        let dst = Array::create_from_array(&pad_input)?;
        dst.lock().unwrap().fill(1.0)?;
        // pad_input.lock().unwrap().copy_to(&dst)?;
        pad_dst = Some(dst);
    }

    // shift kenerl to center it at (0, 0), -1 because we use 0-based index
    let (x_center, y_center, z_center) = {
        let pad_psf = pad_psf.lock().unwrap();
        (
            (pad_psf.width() as i32 / 2) - 1,
            (pad_psf.height() as i32 / 2) - 1,
            (pad_psf.depth() as i32 / 2) - 1,
        )
    };
    pad_psf = tier1::circular_shift(device, &pad_psf, None, x_center, y_center, z_center)?;

    let pad_dst = pad_dst.unwrap();

    // perform deconvolution
    fft_core::perform_deconvolution(
        &pad_input,
        &pad_psf,
        pad_norm.as_ref(),
        Some(pad_dst.clone()),
        iteration as usize,
        regularization,
    )?;

    // unpad the result if needed
    let dst = if padded {
        tier1::unpad(
            device,
            &pad_dst,
            None,
            image_shape[0],
            image_shape[1],
            image_shape[2],
            true,
        )?
    } else {
        pad_dst
    };

    Ok(dst)
}
