use crate::array::{pull, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::tier2;
use crate::tier3;
use crate::types::{DType, BINARY};

pub fn threshold_mean(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let mean_intensity = tier3::mean_of_all_pixels(device, src)?;
    let dst = tier0::create_like(src, dst, BINARY, device)?;
    tier1::greater_constant(device, src, Some(dst), mean_intensity)
}

fn histogram_counts_and_centers(
    device: &DeviceArc,
    src: &ArrayPtr,
) -> Result<(Vec<f64>, Vec<f64>)> {
    const BINS: usize = 256;
    let min_intensity = tier2::minimum_of_all_pixels(device, src)?;
    let max_intensity = tier2::maximum_of_all_pixels(device, src)?;
    let range = (max_intensity - min_intensity) as f64;
    let hist = tier3::histogram(device, src, None, BINS as i32, min_intensity, max_intensity)?;
    let raw_counts: Vec<u32> = pull(&hist)?;
    let counts = raw_counts.into_iter().map(|v| v as f64).collect::<Vec<_>>();
    let bin_centers = (0..BINS)
        .map(|i| (i as f64 * range) / (BINS as f64 - 1.0) + min_intensity as f64)
        .collect::<Vec<_>>();
    Ok((counts, bin_centers))
}

pub fn threshold_otsu(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    const BINS: usize = 256;
    let (counts, bin_centers) = histogram_counts_and_centers(device, src)?;

    let mut weight1 = vec![0.0; BINS];
    let mut running = 0.0;
    for (out, count) in weight1.iter_mut().zip(&counts) {
        running += count;
        *out = running;
    }

    let mut weight2 = vec![0.0; BINS];
    running = 0.0;
    for i in (0..BINS).rev() {
        running += counts[i];
        weight2[i] = running;
    }

    let counts_bin_centers = counts
        .iter()
        .zip(&bin_centers)
        .map(|(count, center)| count * center)
        .collect::<Vec<_>>();
    let mut mean1 = vec![0.0; BINS];
    running = 0.0;
    for i in 0..BINS {
        running += counts_bin_centers[i];
        mean1[i] = running / weight1[i];
    }

    let mut mean2 = vec![0.0; BINS];
    running = 0.0;
    for i in (0..BINS).rev() {
        running += counts_bin_centers[i];
        mean2[i] = running / weight2[i];
    }

    let mut best_idx = 0;
    let mut best_variance = f64::NEG_INFINITY;
    for i in 0..(BINS - 1) {
        let delta = mean1[i] - mean2[i + 1];
        let variance = weight1[i] * weight2[i + 1] * delta * delta;
        if variance > best_variance {
            best_variance = variance;
            best_idx = i;
        }
    }

    let dst = tier0::create_like(src, dst, BINARY, device)?;
    tier1::greater_constant(device, src, Some(dst), bin_centers[best_idx] as f32)
}

pub fn threshold_yen(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    const BINS: usize = 256;
    let min_intensity = tier2::minimum_of_all_pixels(device, src)?;
    let (counts, bin_centers) = histogram_counts_and_centers(device, src)?;
    let dst = tier0::create_like(src, dst, BINARY, device)?;

    let total_counts = counts.iter().sum::<f64>();
    if total_counts <= 0.0 {
        return tier1::greater_constant(device, src, Some(dst), min_intensity);
    }

    let pmf = counts
        .iter()
        .map(|count| count / total_counts)
        .collect::<Vec<_>>();
    let mut p1 = vec![0.0; BINS];
    let mut running = 0.0;
    for i in 0..BINS {
        running += pmf[i];
        p1[i] = running;
    }

    let pmf_squared = pmf.iter().map(|value| value * value).collect::<Vec<_>>();
    let mut p1_sq = vec![0.0; BINS];
    running = 0.0;
    for i in 0..BINS {
        running += pmf_squared[i];
        p1_sq[i] = running;
    }

    let mut p2_sq = vec![0.0; BINS];
    running = 0.0;
    for i in (0..BINS).rev() {
        running += pmf_squared[i];
        p2_sq[i] = running;
    }

    let mut best_idx = 0;
    let mut best_crit = f64::NEG_INFINITY;
    for i in 0..(BINS - 1) {
        let term1 = p1_sq[i] * p2_sq[i + 1];
        let term2 = p1[i] * (1.0 - p1[i]);
        let crit = if term1 > 0.0 && term2 > 0.0 {
            ((term2 * term2) / term1).ln()
        } else {
            f64::NEG_INFINITY
        };
        let crit = if crit.is_finite() {
            crit
        } else {
            f64::NEG_INFINITY
        };
        if crit > best_crit {
            best_crit = crit;
            best_idx = i;
        }
    }

    tier1::greater_constant(device, src, Some(dst), bin_centers[best_idx] as f32)
}

pub fn percentile(device: &DeviceArc, src: &ArrayPtr, percentile: f32) -> Result<f32> {
    const BINS: usize = 256;
    let min_intensity = tier2::minimum_of_all_pixels(device, src)?;
    let max_intensity = tier2::maximum_of_all_pixels(device, src)?;
    let range = (max_intensity - min_intensity) as f64;
    let bin_edges = (0..BINS)
        .map(|i| min_intensity as f64 + (i as f64 * range) / (BINS as f64 - 1.0))
        .collect::<Vec<_>>();
    let hist = tier3::histogram(device, src, None, BINS as i32, min_intensity, max_intensity)?;
    let frequency = pull::<u32>(&hist)?
        .into_iter()
        .map(|v| v as f64)
        .collect::<Vec<_>>();
    let mut cumulative_sum = vec![0.0; frequency.len()];
    let mut running = 0.0;
    for (out, freq) in cumulative_sum.iter_mut().zip(&frequency) {
        running += freq;
        *out = running;
    }
    let target_frequency =
        cumulative_sum.last().copied().unwrap_or(0.0) * (percentile as f64 / 100.0);
    let index = cumulative_sum
        .iter()
        .position(|value| *value >= target_frequency)
        .unwrap_or(cumulative_sum.len().saturating_sub(1));
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
    let fraction = (target_frequency - previous_frequency) / frequency[index];
    let mut result = (lower_edge + fraction * (upper_edge - lower_edge)) as f32;
    if src.lock().unwrap().dtype() != DType::Float {
        result = result.round();
    }
    Ok(result)
}
