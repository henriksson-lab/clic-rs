use crate::array::{Array, ArrayPtr};
use crate::error::{CleError, Result};
use crate::execution::{execute, ParameterValue};
use crate::tier1;
use crate::types::{DType, MType};

type Mat4 = [[f32; 4]; 4];
const AFFINE_TRANSFORM_SRC: &str = include_str!("../kernels/affine_transform.cl");
const AFFINE_TRANSFORM_INTERPOLATE_SRC: &str =
    include_str!("../kernels/affine_transform_interpolate.cl");
const AFFINE_TRANSFORM_DESKEW_X_SRC: &str = include_str!("../kernels/affine_transform_deskew_x.cl");
const AFFINE_TRANSFORM_DESKEW_Y_SRC: &str = include_str!("../kernels/affine_transform_deskew_y.cl");
const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Affine transformation matrix helper.
///
/// Mirrors CLIc's `AffineTransform` class at the API level. The Rust
/// implementation covers the matrix operations needed by translated tier7
/// wrappers.
#[derive(Clone, Debug)]
pub struct AffineTransform {
    matrix: Mat4,
    inverse: Mat4,
    inverse_transpose: Mat4,
    transpose: Mat4,
}

impl AffineTransform {
    pub fn new() -> Self {
        let mut transform = Self {
            matrix: IDENTITY,
            inverse: IDENTITY,
            inverse_transpose: IDENTITY,
            transpose: IDENTITY,
        };
        transform.update();
        transform
    }

    pub fn from_array(array: [f32; 16]) -> Self {
        let mut transform = Self {
            matrix: [
                [array[0], array[1], array[2], array[3]],
                [array[4], array[5], array[6], array[7]],
                [array[8], array[9], array[10], array[11]],
                [array[12], array[13], array[14], array[15]],
            ],
            inverse: IDENTITY,
            inverse_transpose: IDENTITY,
            transpose: IDENTITY,
        };
        transform.update();
        transform
    }

    pub fn scale(&mut self, scale_x: f32, scale_y: f32, scale_z: f32) {
        let mut scale_matrix = IDENTITY;
        scale_matrix[0][0] = scale_x;
        scale_matrix[1][1] = scale_y;
        scale_matrix[2][2] = scale_z;
        self.concats(scale_matrix);
    }

    pub fn rotate(&mut self, axis: usize, angle_deg: f32) {
        let mut rotation_matrix = IDENTITY;
        let (mut angle_sin, mut angle_cos) = Self::deg_to_rad(angle_deg).sin_cos();
        if angle_cos.abs() < f32::EPSILON {
            angle_cos = 0.0;
        }
        if angle_sin.abs() < f32::EPSILON {
            angle_sin = 0.0;
        }
        match axis {
            0 => {
                rotation_matrix[1][1] = angle_cos;
                rotation_matrix[1][2] = -angle_sin;
                rotation_matrix[2][1] = angle_sin;
                rotation_matrix[2][2] = angle_cos;
            }
            1 => {
                rotation_matrix[0][0] = angle_cos;
                rotation_matrix[0][2] = angle_sin;
                rotation_matrix[2][0] = -angle_sin;
                rotation_matrix[2][2] = angle_cos;
            }
            2 => {
                rotation_matrix[0][0] = angle_cos;
                rotation_matrix[0][1] = -angle_sin;
                rotation_matrix[1][0] = angle_sin;
                rotation_matrix[1][1] = angle_cos;
            }
            _ => panic!("Invalid axis"),
        }
        self.concats(rotation_matrix);
    }

    pub fn rotate_around_x_axis(&mut self, angle_deg: f32) {
        self.rotate(0, angle_deg);
    }

    pub fn rotate_around_y_axis(&mut self, angle_deg: f32) {
        self.rotate(1, angle_deg);
    }

    pub fn rotate_around_z_axis(&mut self, angle_deg: f32) {
        self.rotate(2, angle_deg);
    }

    pub fn translate(&mut self, translate_x: f32, translate_y: f32, translate_z: f32) {
        let mut translation_matrix = IDENTITY;
        translation_matrix[0][3] = translate_x;
        translation_matrix[1][3] = translate_y;
        translation_matrix[2][3] = translate_z;
        self.concats(translation_matrix);
    }

    pub fn center(&mut self, shape: [usize; 3], undo: bool) {
        let presign = if undo { 1.0 } else { -1.0 };
        let centering_x = if shape[0] != 1 {
            presign * shape[0] as f32 / 2.0
        } else {
            0.0
        };
        let centering_y = if shape[1] != 1 {
            presign * shape[1] as f32 / 2.0
        } else {
            0.0
        };
        let centering_z = if shape[2] != 1 {
            presign * shape[2] as f32 / 2.0
        } else {
            0.0
        };
        self.translate(centering_x, centering_y, centering_z);
    }

    pub fn shear_in_z_plane(&mut self, shear_x_deg: f32, shear_y_deg: f32) {
        if !(-90.0..=90.0).contains(&shear_x_deg) {
            panic!("Shear X angle must be between -90 and 90 degrees");
        }
        if !(-90.0..=90.0).contains(&shear_y_deg) {
            panic!("Shear Y angle must be between -90 and 90 degrees");
        }

        let mut matrix = IDENTITY;
        matrix[0][1] = Self::shear_angle_to_shear_factor(shear_x_deg);
        matrix[1][0] = Self::shear_angle_to_shear_factor(shear_y_deg);
        self.pre_concats(matrix);
    }

    pub fn shear_in_y_plane(&mut self, shear_x_deg: f32, shear_z_deg: f32) {
        if !(-90.0..=90.0).contains(&shear_x_deg) {
            panic!("Shear X angle must be between -90 and 90 degrees");
        }
        if !(-90.0..=90.0).contains(&shear_z_deg) {
            panic!("Shear Z angle must be between -90 and 90 degrees");
        }

        let mut matrix = IDENTITY;
        matrix[0][2] = Self::shear_angle_to_shear_factor(shear_x_deg);
        matrix[2][0] = Self::shear_angle_to_shear_factor(shear_z_deg);
        self.pre_concats(matrix);
    }

    pub fn shear_in_x_plane(&mut self, shear_y_deg: f32, shear_z_deg: f32) {
        if !(-90.0..=90.0).contains(&shear_y_deg) {
            panic!("Shear Y angle must be between -90 and 90 degrees");
        }
        if !(-90.0..=90.0).contains(&shear_z_deg) {
            panic!("Shear Z angle must be between -90 and 90 degrees");
        }

        let mut matrix = IDENTITY;
        matrix[1][2] = Self::shear_angle_to_shear_factor(shear_y_deg);
        matrix[2][1] = Self::shear_angle_to_shear_factor(shear_z_deg);
        self.pre_concats(matrix);
    }

    pub fn deskew_x(
        &mut self,
        angle_deg: f32,
        voxel_size_x: f32,
        _voxel_size_y: f32,
        voxel_size_z: f32,
        scale_factor: f32,
    ) {
        let pi180 = std::f32::consts::PI / 180.0;
        let shear_factor = ((90.0 - angle_deg) * pi180).sin() * (voxel_size_z / voxel_size_x);
        self.matrix[0][2] += shear_factor;
        let new_dz = (angle_deg * pi180).sin() * voxel_size_z;
        let scale_factor_z = (new_dz / voxel_size_x) * scale_factor;
        self.scale(scale_factor, scale_factor, scale_factor_z);
        self.rotate_around_y_axis(angle_deg);
    }

    pub fn deskew_y(
        &mut self,
        angle_deg: f32,
        _voxel_size_x: f32,
        voxel_size_y: f32,
        voxel_size_z: f32,
        scale_factor: f32,
    ) {
        let pi180 = std::f32::consts::PI / 180.0;
        let shear_factor = ((90.0 - angle_deg) * pi180).sin() * (voxel_size_z / voxel_size_y);
        self.matrix[1][2] += shear_factor;
        let new_dz = (angle_deg * pi180).sin() * voxel_size_z;
        let scale_factor_z = (new_dz / voxel_size_y) * scale_factor;
        self.scale(scale_factor, scale_factor, scale_factor_z);
        self.rotate_around_x_axis(0.0 - angle_deg);
    }

    pub fn get_matrix(&self) -> Mat4 {
        self.matrix
    }

    pub fn get_inverse(&self) -> Mat4 {
        self.inverse
    }

    pub fn get_transpose(&self) -> Mat4 {
        self.transpose
    }

    pub fn get_inverse_transpose(&self) -> Mat4 {
        self.inverse_transpose
    }

    pub fn to_array(matrix: Mat4) -> [f32; 16] {
        [
            matrix[0][0],
            matrix[1][0],
            matrix[2][0],
            matrix[3][0],
            matrix[0][1],
            matrix[1][1],
            matrix[2][1],
            matrix[3][1],
            matrix[0][2],
            matrix[1][2],
            matrix[2][2],
            matrix[3][2],
            matrix[0][3],
            matrix[1][3],
            matrix[2][3],
            matrix[3][3],
        ]
    }

    fn update(&mut self) {
        let original = self.matrix;
        let mut matrix = original;
        let mut inv = IDENTITY;
        for col in 0..4 {
            let mut pivot = col;
            let mut pivot_abs = matrix[col][col].abs();
            for row in (col + 1)..4 {
                let candidate_abs = matrix[row][col].abs();
                if candidate_abs > pivot_abs {
                    pivot = row;
                    pivot_abs = candidate_abs;
                }
            }
            if pivot != col {
                matrix.swap(col, pivot);
                inv.swap(col, pivot);
            }

            let pivot_value = matrix[col][col];
            for k in 0..4 {
                matrix[col][k] /= pivot_value;
                inv[col][k] /= pivot_value;
            }
            for row in 0..4 {
                if row == col {
                    continue;
                }
                let factor = matrix[row][col];
                for k in 0..4 {
                    matrix[row][k] -= factor * matrix[col][k];
                    inv[row][k] -= factor * inv[col][k];
                }
            }
        }

        let mut transpose = [[0.0; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                transpose[row][col] = original[col][row];
            }
        }

        let mut inverse_transpose = [[0.0; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                inverse_transpose[row][col] = inv[col][row];
            }
        }

        self.inverse = inv;
        self.inverse_transpose = inverse_transpose;
        self.transpose = transpose;
    }

    fn concats(&mut self, matrix: Mat4) {
        let right = self.matrix;
        let mut out = [[0.0; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                out[row][col] = matrix[row][0] * right[0][col]
                    + matrix[row][1] * right[1][col]
                    + matrix[row][2] * right[2][col]
                    + matrix[row][3] * right[3][col];
            }
        }
        self.matrix = out;
        self.update();
    }

    fn pre_concats(&mut self, matrix: Mat4) {
        let left = self.matrix;
        let mut out = [[0.0; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                out[row][col] = left[row][0] * matrix[0][col]
                    + left[row][1] * matrix[1][col]
                    + left[row][2] * matrix[2][col]
                    + left[row][3] * matrix[3][col];
            }
        }
        self.matrix = out;
        self.update();
    }
}

impl AffineTransform {
    fn deg_to_rad(angle_deg: f32) -> f32 {
        angle_deg * (std::f32::consts::PI / 180.0)
    }

    fn shear_angle_to_shear_factor(angle_deg: f32) -> f32 {
        1.0 / Self::deg_to_rad(90.0 - angle_deg).tan()
    }
}

impl Drop for AffineTransform {
    fn drop(&mut self) {}
}

/// Prepare output shape and transform.
pub fn prepare_output_shape_and_transform(
    src: &ArrayPtr,
    transform: &AffineTransform,
) -> (usize, usize, usize, AffineTransform) {
    let src = src.lock().unwrap();
    let width = src.width() as f32;
    let height = src.height() as f32;
    let depth = src.depth() as f32;
    drop(src);

    let bbox = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, depth, 1.0],
        [0.0, height, 0.0, 1.0],
        [width, 0.0, 0.0, 1.0],
        [width, height, 0.0, 1.0],
        [0.0, height, depth, 1.0],
        [width, 0.0, depth, 1.0],
        [width, height, depth, 1.0],
    ];

    let matrix = transform.get_matrix();
    let mut updated_bbox = [[0.0; 4]; 8];
    for index in 0..bbox.len() {
        let point = bbox[index];
        updated_bbox[index] = [
            matrix[0][0] * point[0]
                + matrix[0][1] * point[1]
                + matrix[0][2] * point[2]
                + matrix[0][3] * point[3],
            matrix[1][0] * point[0]
                + matrix[1][1] * point[1]
                + matrix[1][2] * point[2]
                + matrix[1][3] * point[3],
            matrix[2][0] * point[0]
                + matrix[2][1] * point[1]
                + matrix[2][2] * point[2]
                + matrix[2][3] * point[3],
            matrix[3][0] * point[0]
                + matrix[3][1] * point[1]
                + matrix[3][2] * point[2]
                + matrix[3][3] * point[3],
        ];
    }

    let mut min = updated_bbox[0];
    let mut max = min;
    for point in updated_bbox {
        for axis in 0..4 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }

    let mut update_transform = transform.clone();
    let width = (max[0] - min[0]).round() as usize;
    let height = (max[1] - min[1]).round() as usize;
    let depth = (max[2] - min[2]).round() as usize;
    update_transform.translate(-min[0], -min[1], -min[2]);

    (width, height, depth, update_transform)
}

/// Apply an affine transform to an array.
///
/// Mirrors CLIc's `apply_affine_transform`.
pub fn apply_affine_transform(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    transform: &AffineTransform,
    interpolate: bool,
    auto_resize: bool,
) -> Result<ArrayPtr> {
    let (device, dim, src_dtype, mtype, src_width, src_height, src_depth) = {
        let src = src.lock().unwrap();
        (
            src.device().clone(),
            src.dimension(),
            src.dtype(),
            src.mtype(),
            src.width(),
            src.height(),
            src.depth(),
        )
    };

    let mut new_transform = transform.clone();
    let mut width = src_width;
    let mut height = src_height;
    let mut depth = src_depth;

    if auto_resize {
        (width, height, depth, new_transform) = prepare_output_shape_and_transform(src, transform);
    }

    let dst = match dst {
        Some(dst) => dst,
        None => {
            let dtype = if interpolate { DType::Float } else { src_dtype };
            Array::create(width, height, depth, dim, dtype, mtype, &device)?
        }
    };
    let inverse_transpose = AffineTransform::to_array(new_transform.get_inverse_transpose());
    let mat = Array::create_with_data(4, 4, 1, 2, MType::Buffer, &inverse_transpose, &device)?;

    let mut image = src.clone();
    if interpolate && mtype != MType::Image {
        match Array::create(
            src_width,
            src_height,
            src_depth,
            dim,
            DType::Float,
            MType::Image,
            &device,
        )
        .and_then(|created| {
            tier1::copy(&device, src, Some(created.clone()))?;
            Ok(created)
        }) {
            Ok(created) => image = created,
            Err(_) => {
                let platform = device.get_platform();
                if platform == "CUDA" || platform == "NVIDIA" {
                    eprintln!(
                        "Warning: Interpolated transform is not implemented with the CUDA backend."
                    );
                } else {
                    eprintln!(
                        "Warning: Device does not support Image type required for interpolation."
                    );
                }
                eprintln!("-> We fall back to non-interpolated transform.");
            }
        }
    }

    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(image.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
        ("mat", ParameterValue::Array(mat)),
    ];
    let kernel = if interpolate && image.lock().unwrap().mtype() == MType::Image {
        (
            "affine_transform_interpolate",
            AFFINE_TRANSFORM_INTERPOLATE_SRC,
        )
    } else {
        ("affine_transform", AFFINE_TRANSFORM_SRC)
    };
    execute(&device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

/// Apply a 3D deskew transform.
///
/// Mirrors CLIc's `apply_affine_transform_deskew_3d`.
pub fn apply_affine_transform_deskew_3d(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    transform: &AffineTransform,
    deskewing_angle: f32,
    voxel_size_x: f32,
    voxel_size_y: f32,
    voxel_size_z: f32,
    deskew_direction: i32,
    auto_resize: bool,
) -> Result<ArrayPtr> {
    if src.lock().unwrap().depth() == 1 {
        return Err(CleError::Other(
            "Deskewing is only available for 3D images.".to_string(),
        ));
    }
    let (device, dim, dtype, src_mtype, src_width, src_height, src_depth) = {
        let src = src.lock().unwrap();
        (
            src.device().clone(),
            src.dimension(),
            src.dtype(),
            src.mtype(),
            src.width(),
            src.height(),
            src.depth(),
        )
    };

    let mut new_transform = transform.clone();
    let mut width = src_width;
    let mut height = src_height;
    let mut depth = src_depth;

    if auto_resize {
        (width, height, depth, new_transform) = prepare_output_shape_and_transform(src, transform);
    }

    let dst = match dst {
        Some(dst) => dst,
        None => Array::create(width, height, depth, dim, dtype, src_mtype, &device)?,
    };

    let image = if src_mtype == MType::Image {
        src.clone()
    } else {
        match Array::create(
            src_width,
            src_height,
            src_depth,
            dim,
            dtype,
            MType::Image,
            &device,
        )
        .and_then(|created| {
            src.lock().unwrap().copy_to(&created)?;
            Ok(created)
        }) {
            Ok(created) => created,
            Err(_) => {
                eprintln!(
                    "Warning: Device does not support Image type. Deskewing is not available, falling back to non-deskewed transform."
                );
                return apply_affine_transform(src, Some(dst), &new_transform, false, false);
            }
        }
    };

    let inverse_transpose = AffineTransform::to_array(new_transform.get_inverse_transpose());
    let mat = Array::create_with_data(4, 4, 1, 2, MType::Buffer, &inverse_transpose, &device)?;

    let tantheta = (deskewing_angle * std::f32::consts::PI / 180.0).tan();
    let sintheta = (deskewing_angle * std::f32::consts::PI / 180.0).sin();
    let costheta = (deskewing_angle * std::f32::consts::PI / 180.0).cos();

    let mut pixel_step = 0.0;
    let mut kernel = ("", "");
    match deskew_direction {
        0 => {
            kernel = ("affine_transform_deskew_x", AFFINE_TRANSFORM_DESKEW_X_SRC);
            pixel_step = voxel_size_z / voxel_size_y;
        }
        1 => {
            kernel = ("affine_transform_deskew_y", AFFINE_TRANSFORM_DESKEW_Y_SRC);
            pixel_step = voxel_size_z / voxel_size_x;
        }
        _ => {}
    }

    let range = {
        let dst = dst.lock().unwrap();
        [dst.width(), dst.height(), dst.depth()]
    };
    let params = vec![
        ("src", ParameterValue::Array(image)),
        ("dst", ParameterValue::Array(dst.clone())),
        ("mat", ParameterValue::Array(mat)),
        ("pixel_step", ParameterValue::Float(pixel_step)),
        ("tantheta", ParameterValue::Float(tantheta)),
        ("costheta", ParameterValue::Float(costheta)),
        ("sintheta", ParameterValue::Float(sintheta)),
    ];
    execute(&device, kernel, &params, range, [0, 0, 0], &[])?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn deskew_transform_rejects_2d_inputs_before_backend_work() {
        let src = Arc::new(Mutex::new(Array {
            width: 8,
            height: 8,
            depth: 1,
            dim: crate::utils::shape_to_dimension(8, 8, 1),
            dtype: DType::Float,
            mtype: MType::Buffer,
            device: Arc::new(DummyDevice),
            mem: None,
            owns_memory: true,
        }));
        let err = match apply_affine_transform_deskew_3d(
            &src,
            None,
            &AffineTransform::new(),
            45.0,
            1.0,
            1.0,
            1.0,
            0,
            true,
        ) {
            Ok(_) => panic!("2D deskew unexpectedly succeeded"),
            Err(err) => err,
        };

        assert_eq!(
            err.to_string(),
            "Deskewing is only available for 3D images."
        );
    }
}
