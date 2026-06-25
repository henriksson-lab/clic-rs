use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::tier3;
use crate::types::{DType, MType, BINARY};

pub fn threshold_mean(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let mean_intensity = tier3::mean_of_all_pixels(device, src)?;
    let dst = tier0::create_like(src, dst, BINARY, device)?;
    tier1::greater_constant(device, src, Some(dst.clone()), mean_intensity)?;
    Ok(dst)
}

pub fn threshold_otsu(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    // Initialize histogram
    const BIN: usize = 256;
    let min_intensity = tier2::minimum_of_all_pixels(device, src)?;
    let max_intensity = tier2::maximum_of_all_pixels(device, src)?;
    let range = (max_intensity - min_intensity) as f64;
    let src_device = src.lock().unwrap().device().clone();

    // Compute histogram
    let hist_array = Array::create(BIN, 1, 1, 1, DType::Float, MType::Buffer, &src_device)?;
    tier3::histogram(
        device,
        src,
        Some(hist_array.clone()),
        BIN as i32,
        min_intensity,
        max_intensity,
    )?;
    let mut counts = vec![0.0_f32; BIN];
    hist_array.lock().unwrap().read_to(&mut counts)?;

    // Compute bin centers
    let mut bin_centers = vec![0.0; BIN];
    for i in 0..BIN {
        bin_centers[i] = i as f64;
    }
    for i in 0..BIN {
        bin_centers[i] = (bin_centers[i] * range) / (BIN - 1) as f64 + min_intensity as f64;
    }

    // Compute weight1
    let mut weight1 = vec![0.0; BIN];
    let mut weight2 = vec![0.0; BIN];
    let mut mean1 = vec![0.0; BIN];
    let mut mean2 = vec![0.0; BIN];
    let mut variance12 = vec![0.0; BIN - 1];
    let mut running_counts = 0.0_f32;
    for i in 0..BIN {
        running_counts += counts[i];
        weight1[i] = running_counts as f64;
    }

    // Compute weight2
    let reversed_counts = counts.iter().rev().copied().collect::<Vec<_>>();
    running_counts = 0.0;
    for i in 0..BIN {
        running_counts += reversed_counts[i];
        weight2[BIN - 1 - i] = running_counts as f64;
    }

    // Compute mean1
    let mut counts_bin_centers = vec![0.0; BIN];
    for i in 0..BIN {
        counts_bin_centers[i] = counts[i] as f64 * bin_centers[i];
    }
    let mut running = 0.0;
    for i in 0..BIN {
        running += counts_bin_centers[i];
        mean1[i] = running / weight1[i];
    }

    // Compute mean2
    running = 0.0;
    for i in (0..BIN).rev() {
        running += counts_bin_centers[i];
        mean2[i] = running / weight2[i];
    }

    // Compute variance12
    for i in 0..(BIN - 1) {
        variance12[i] =
            weight1[i] * weight2[i + 1] * (mean1[i] - mean2[i + 1]) * (mean1[i] - mean2[i + 1]);
    }

    // Find the maximum variance and threshold value associated with it
    let mut best_idx = 0_usize;
    let mut best_variance = variance12[0];
    for i in 1..variance12.len() {
        if best_variance < variance12[i] {
            best_variance = variance12[i];
            best_idx = i;
        }
    }
    let threshold = bin_centers[best_idx];

    // Create binary image with threshold
    let dst = tier0::create_like(src, dst, BINARY, device)?;
    tier1::greater_constant(device, src, Some(dst.clone()), threshold as f32)?;
    Ok(dst)
}

pub fn threshold_yen(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, BINARY, device)?;

    // Initialize histogram
    const BIN: usize = 256;
    let min_intensity = tier2::minimum_of_all_pixels(device, src)?;
    let max_intensity = tier2::maximum_of_all_pixels(device, src)?;
    let range = (max_intensity - min_intensity) as f64;
    let src_device = src.lock().unwrap().device().clone();

    // Compute histogram
    let hist_array = Array::create(BIN, 1, 1, 1, DType::Float, MType::Buffer, &src_device)?;
    tier3::histogram(
        device,
        src,
        Some(hist_array.clone()),
        BIN as i32,
        min_intensity,
        max_intensity,
    )?;
    let mut counts = vec![0.0_f32; BIN];
    hist_array.lock().unwrap().read_to(&mut counts)?;

    // Compute bin centers
    let mut bin_centers = vec![0.0; BIN];
    for i in 0..BIN {
        bin_centers[i] = i as f64;
    }
    for i in 0..BIN {
        bin_centers[i] = (bin_centers[i] * range) / (BIN - 1) as f64 + min_intensity as f64;
    }

    // pmf = counts.astype('float32', copy=False) / counts.sum()
    let mut total_counts = 0.0;
    for i in 0..BIN {
        total_counts += counts[i] as f64;
    }
    if total_counts <= 0.0 {
        tier1::greater_constant(device, src, Some(dst.clone()), min_intensity)?;
        return Ok(dst);
    }

    let mut pmf = vec![0.0; BIN];
    for i in 0..BIN {
        pmf[i] = counts[i] as f64 / total_counts;
    }

    // P1 = np.cumsum(pmf)
    let mut p1 = vec![0.0; BIN];
    let mut running = 0.0;
    for i in 0..BIN {
        running += pmf[i];
        p1[i] = running;
    }

    // P1_sq = np.cumsum(pmf**2)
    let mut pmf_squared = vec![0.0; BIN];
    for i in 0..BIN {
        pmf_squared[i] = pmf[i] * pmf[i];
    }
    let mut p1_sq = vec![0.0; BIN];
    running = 0.0;
    for i in 0..BIN {
        running += pmf_squared[i];
        p1_sq[i] = running;
    }

    // P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    let reversed_pmf_squared = pmf_squared.iter().rev().copied().collect::<Vec<_>>();
    let mut p2_sq = vec![0.0; BIN];
    running = 0.0;
    for i in 0..BIN {
        running += reversed_pmf_squared[i];
        p2_sq[BIN - 1 - i] = running;
    }

    // crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    let mut crit = vec![0.0; BIN - 1];
    let negative_infinity = -f64::INFINITY;
    for i in 0..(BIN - 1) {
        let term1 = p1_sq[i] * p2_sq[i + 1];
        let term2 = p1[i] * (1.0 - p1[i]);
        if term1 > 0.0 && term2 > 0.0 {
            let value = (term2 * term2) / term1;
            let logv = value.ln();
            crit[i] = if logv.is_finite() {
                logv
            } else {
                negative_infinity
            };
        } else {
            crit[i] = negative_infinity;
        }
    }

    // bin_centers[crit.argmax()]
    let mut best_idx = 0_usize;
    let mut best_crit = crit[0];
    for i in 1..crit.len() {
        if best_crit < crit[i] {
            best_crit = crit[i];
            best_idx = i;
        }
    }
    let threshold = bin_centers[best_idx];

    // Create binary image with threshold
    tier1::greater_constant(device, src, Some(dst.clone()), threshold as f32)?;
    Ok(dst)
}

pub fn percentile(device: &DeviceArc, src: &ArrayPtr, percentile: f32) -> Result<f32> {
    // Initialize histogram
    const BIN: usize = 256;
    let min_intensity = tier2::minimum_of_all_pixels(device, src)?;
    let max_intensity = tier2::maximum_of_all_pixels(device, src)?;
    let range = (max_intensity - min_intensity) as f64;
    let src_device = src.lock().unwrap().device().clone();

    // compute bin edges
    let mut bin_edges = vec![0.0; BIN];
    for i in 0..BIN {
        bin_edges[i] = i as f64;
    }
    for i in 0..BIN {
        bin_edges[i] = min_intensity as f64 + (bin_edges[i] * range) / (BIN as f64 - 1.0);
    }

    // Compute histogram
    let hist_array = Array::create(BIN, 1, 1, 1, DType::Float, MType::Buffer, &src_device)?;
    tier3::histogram(
        device,
        src,
        Some(hist_array.clone()),
        BIN as i32,
        min_intensity,
        max_intensity,
    )?;
    let mut frequency = vec![0.0_f32; BIN];
    hist_array.lock().unwrap().read_to(&mut frequency)?;

    // compute cumulative sum of the vector frequency
    let mut cumulative_sum = vec![0.0; frequency.len()];
    let mut running = 0.0_f32;
    for i in 0..frequency.len() {
        running += frequency[i];
        cumulative_sum[i] = running as f64;
    }

    // Calculate total frequency and target frequency
    let total_frequency = cumulative_sum.last().copied().unwrap_or(0.0);
    let target_frequency = total_frequency * (percentile as f64 / 100.0);

    // Find the bin containing the target frequency using binary search
    let index = cumulative_sum.partition_point(|value| *value < target_frequency);

    // Compute the percentile value
    let lower_edge = if index == 0 {
        bin_edges[index]
    } else {
        bin_edges[index - 1]
    };
    let upper_edge = bin_edges[index];
    let previous_frequency = if index == 0 {
        0.0
    } else {
        cumulative_sum[index - 1]
    };
    let fraction = (target_frequency - previous_frequency) / frequency[index] as f64;
    let mut result = (lower_edge + fraction * (upper_edge - lower_edge)) as f32;
    if src.lock().unwrap().dtype() != DType::Float {
        result = result.round();
    }
    Ok(result)
}
