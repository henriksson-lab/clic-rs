use std::time::{SystemTime, UNIX_EPOCH};

use crate::array::{Array, ArrayPtr};
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier2;
use crate::types::{MType, LABEL};
use crate::utils::shape_to_dimension;

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9e37_79b9_7f4a_7c15);
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let value = (self.state >> 40) as u32;
        (value as f32 + 1.0) / ((1_u32 << 24) as f32 + 1.0)
    }

    fn normal(&mut self, sigma: f32) -> f32 {
        if sigma == 0.0 {
            return 0.0;
        }
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

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
    let mut rng = SimpleRng::new();
    let mut all_x = Vec::new();
    let mut all_y = Vec::new();
    let mut all_z = Vec::new();

    let mut z = 0.0;
    while z < depth as f32 {
        let mut row = 0;
        let mut y = 0.0;
        while y < height as f32 {
            let offset_x = if row % 2 != 0 { delta_x / 2.0 } else { 0.0 };
            let mut x = offset_x;
            while x < width as f32 {
                all_x.push(x + if width > 1 { rng.normal(sigma_x) } else { 0.0 });
                all_y.push(y + if height > 1 { rng.normal(sigma_y) } else { 0.0 });
                all_z.push(z + if depth > 1 { rng.normal(sigma_z) } else { 0.0 });
                x += delta_x;
            }
            row += 1;
            y += delta_y;
        }
        z += delta_z;
    }

    (all_x, all_y, all_z)
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

    let point_count = x_coords.len();
    let mut point_list_data = vec![0.0_f32; point_count * 3];
    point_list_data[..point_count].copy_from_slice(&x_coords);
    point_list_data[point_count..2 * point_count].copy_from_slice(&y_coords);
    point_list_data[2 * point_count..].copy_from_slice(&z_coords);
    let point_list = Array::create_with_data(
        point_count,
        3,
        1,
        2,
        MType::Buffer,
        &point_list_data,
        device,
    )?;
    let centroids = Array::create(width, height, depth, dim, LABEL, MType::Buffer, device)?;
    tier2::pointlist_to_labelled_spots(device, &point_list, Some(centroids.clone()))?;
    tier2::extend_labeling_via_voronoi(device, &centroids, Some(dst))
}
