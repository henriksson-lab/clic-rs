use std::time::{SystemTime, UNIX_EPOCH};

use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier2;
use crate::types::{DType, MType, LABEL};
use crate::utils::shape_to_dimension;

#[allow(clippy::too_many_arguments)]
fn coordinate_generator(
    width: usize,
    height: usize,
    depth: usize,
    delta_x: f32,
    delta_y: f32,
    delta_z: f32,
    sigma_x: f32,
    sigma_y: f32,
    sigma_z: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut all_x_coords = Vec::new();
    let mut all_y_coords = Vec::new();
    let mut all_z_coords = Vec::new();

    let mut rng_state = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9e37_79b9_7f4a_7c15);
    let mut normal = |sigma: f32| -> f32 {
        if sigma == 0.0 {
            return 0.0;
        }
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1_value = (rng_state >> 40) as u32;
        let u1 = ((u1_value as f32 + 1.0) / ((1_u32 << 24) as f32 + 1.0)).max(f32::MIN_POSITIVE);
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2_value = (rng_state >> 40) as u32;
        let u2 = (u2_value as f32 + 1.0) / ((1_u32 << 24) as f32 + 1.0);
        sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    };

    let mut z = 0.0;
    let mut _k = 0;
    while z < depth as f32 {
        let mut i = 0;
        let mut y = 0.0;
        while y < height as f32 {
            let offset_x = if i % 2 != 0 { delta_x / 2.0 } else { 0.0 };

            let mut x_coords = Vec::new();
            let mut x = offset_x;
            while x < width as f32 {
                x_coords.push(x);
                x += delta_x;
            }

            let num_coords = x_coords.len();
            let mut y_coords = vec![y; num_coords];
            let mut z_coords = vec![z; num_coords];

            for j in 0..num_coords {
                x_coords[j] += if width > 1 { normal(sigma_x) } else { 0.0 };
                y_coords[j] += if height > 1 { normal(sigma_y) } else { 0.0 };
                z_coords[j] += if depth > 1 { normal(sigma_z) } else { 0.0 };
            }

            all_x_coords.extend(x_coords);
            all_y_coords.extend(y_coords);
            all_z_coords.extend(z_coords);

            i += 1;
            y += delta_y;
        }
        _k += 1;
        z += delta_z;
    }

    (all_x_coords, all_y_coords, all_z_coords)
}

#[allow(clippy::too_many_arguments)]
pub fn artificial_tissue(
    device: &DeviceArc,
    width: usize,
    height: usize,
    depth: usize,
    delta_x: f32,
    delta_y: f32,
    delta_z: f32,
    sigma_x: f32,
    sigma_y: f32,
    sigma_z: f32,
) -> Result<ArrayPtr> {
    let dim = shape_to_dimension(width, height, depth);
    let dst = Array::create(width, height, depth, dim, LABEL, MType::Buffer, device)?;

    let (mut x_coords, mut y_coords, mut z_coords) = coordinate_generator(
        width, height, depth, delta_x, delta_y, delta_z, sigma_x, sigma_y, sigma_z,
    );
    for i in 0..x_coords.len() {
        x_coords[i] = x_coords[i].clamp(0.0, (width - 1) as f32);
        y_coords[i] = y_coords[i].clamp(0.0, (height - 1) as f32);
        z_coords[i] = z_coords[i].clamp(0.0, (depth - 1) as f32);
    }

    let nb_points = x_coords.len();
    let point_list = Array::create(nb_points, 3, 1, 2, DType::Float, MType::Buffer, device)?;
    point_list
        .lock()
        .unwrap()
        .write_from_region(&x_coords, [nb_points, 1, 1], [0, 0, 0])?;
    point_list
        .lock()
        .unwrap()
        .write_from_region(&y_coords, [nb_points, 1, 1], [0, 1, 0])?;
    point_list
        .lock()
        .unwrap()
        .write_from_region(&z_coords, [nb_points, 1, 1], [0, 2, 0])?;
    let centroids = Array::create_from_array(&dst)?;
    tier2::pointlist_to_labelled_spots(device, &point_list, Some(centroids.clone()))?;
    tier2::extend_labeling_via_voronoi(device, &centroids, Some(dst))
}
