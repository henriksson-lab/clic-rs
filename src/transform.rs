use crate::array::ArrayPtr;
use crate::error::Result;
use crate::tier7;

type Mat4 = [[f32; 4]; 4];

/// Affine transformation matrix helper.
///
/// Mirrors CLIc's `AffineTransform` class at the API level. The Rust
/// implementation covers the matrix operations needed by translated tier7
/// wrappers. Dedicated resize/interpolation/deskew kernels are still handled as
/// follow-up backend work.
#[derive(Clone, Debug)]
pub struct AffineTransform {
    matrix: Mat4,
}

impl AffineTransform {
    pub fn new() -> Self {
        Self { matrix: identity() }
    }

    pub fn from_array(array: [f32; 16]) -> Self {
        Self {
            matrix: [
                [array[0], array[1], array[2], array[3]],
                [array[4], array[5], array[6], array[7]],
                [array[8], array[9], array[10], array[11]],
                [array[12], array[13], array[14], array[15]],
            ],
        }
    }

    pub fn scale(&mut self, scale_x: f32, scale_y: f32, scale_z: f32) {
        self.concats(scale_matrix(scale_x, scale_y, scale_z));
    }

    pub fn rotate(&mut self, axis: usize, angle_deg: f32) {
        let rotation = match axis {
            0 => rotation_x(angle_deg),
            1 => rotation_y(angle_deg),
            2 => rotation_z(angle_deg),
            _ => return,
        };
        self.concats(rotation);
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
        self.concats(translation(translate_x, translate_y, translate_z));
    }

    pub fn center(&mut self, shape: [usize; 3], undo: bool) {
        let presign = if undo { 1.0 } else { -1.0 };
        let x = if shape[0] != 1 {
            presign * shape[0] as f32 / 2.0
        } else {
            0.0
        };
        let y = if shape[1] != 1 {
            presign * shape[1] as f32 / 2.0
        } else {
            0.0
        };
        let z = if shape[2] != 1 {
            presign * shape[2] as f32 / 2.0
        } else {
            0.0
        };
        self.translate(x, y, z);
    }

    pub fn shear_in_z_plane(&mut self, shear_x_deg: f32, shear_y_deg: f32) {
        let mut matrix = identity();
        matrix[0][1] = shear_angle_to_shear_factor(shear_x_deg);
        matrix[1][0] = shear_angle_to_shear_factor(shear_y_deg);
        self.pre_concats(matrix);
    }

    pub fn shear_in_y_plane(&mut self, shear_x_deg: f32, shear_z_deg: f32) {
        let mut matrix = identity();
        matrix[0][2] = shear_angle_to_shear_factor(shear_x_deg);
        matrix[2][0] = shear_angle_to_shear_factor(shear_z_deg);
        self.pre_concats(matrix);
    }

    pub fn shear_in_x_plane(&mut self, shear_y_deg: f32, shear_z_deg: f32) {
        let mut matrix = identity();
        matrix[1][2] = shear_angle_to_shear_factor(shear_y_deg);
        matrix[2][1] = shear_angle_to_shear_factor(shear_z_deg);
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
        let shear_factor = (90.0 - angle_deg).to_radians().sin() * (voxel_size_z / voxel_size_x);
        let scale_factor_z =
            angle_deg.to_radians().sin() * voxel_size_z / voxel_size_x * scale_factor;
        let mut shear = identity();
        shear[0][2] = shear_factor;
        self.concats(shear);
        self.concats(scale_matrix(scale_factor, scale_factor, scale_factor_z));
        self.concats(rotation_y(angle_deg));
    }

    pub fn deskew_y(
        &mut self,
        angle_deg: f32,
        _voxel_size_x: f32,
        voxel_size_y: f32,
        voxel_size_z: f32,
        scale_factor: f32,
    ) {
        let shear_factor = (90.0 - angle_deg).to_radians().sin() * (voxel_size_z / voxel_size_y);
        let scale_factor_z =
            angle_deg.to_radians().sin() * voxel_size_z / voxel_size_y * scale_factor;
        let mut shear = identity();
        shear[1][2] = shear_factor;
        self.concats(shear);
        self.concats(scale_matrix(scale_factor, scale_factor, scale_factor_z));
        self.concats(rotation_x(-angle_deg));
    }

    pub fn get_matrix(&self) -> Mat4 {
        self.matrix
    }

    pub fn get_inverse(&self) -> Mat4 {
        invert_affine_matrix(self.matrix)
    }

    pub fn get_transpose(&self) -> Mat4 {
        transpose(self.matrix)
    }

    pub fn get_inverse_transpose(&self) -> Mat4 {
        transpose(self.get_inverse())
    }

    pub fn to_array(matrix: Mat4) -> [f32; 16] {
        flatten(matrix)
    }

    fn concats(&mut self, matrix: Mat4) {
        self.matrix = mul(matrix, self.matrix);
    }

    fn pre_concats(&mut self, matrix: Mat4) {
        self.matrix = mul(self.matrix, matrix);
    }
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self::new()
    }
}

/// Prepare output shape and transform.
///
/// CLIc can resize the output to fit a transformed bounding box. The current
/// Rust affine helper keeps the input shape, so this returns the original shape
/// and cloned transform.
pub fn prepare_output_shape_and_transform(
    src: &ArrayPtr,
    transform: &AffineTransform,
) -> (usize, usize, usize, AffineTransform) {
    let src = src.lock().unwrap();
    (src.width(), src.height(), src.depth(), transform.clone())
}

/// Apply an affine transform to an array.
///
/// Mirrors CLIc's `apply_affine_transform`; `interpolate` and `resize` are
/// accepted for API parity but ignored by the current Rust affine kernel path.
pub fn apply_affine_transform(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    transform: &AffineTransform,
    _interpolate: bool,
    _resize: bool,
) -> Result<ArrayPtr> {
    let device = src.lock().unwrap().device().clone();
    let inverse = AffineTransform::to_array(transform.get_inverse());
    tier7::affine_transform(&device, src, dst, &inverse)
}

/// Apply a 3D deskew transform.
///
/// Mirrors CLIc's `apply_affine_transform_deskew_3d`; the Rust version falls
/// back to the generic affine kernel until dedicated deskew kernels are ported.
pub fn apply_affine_transform_deskew_3d(
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    transform: &AffineTransform,
    _deskewing_angle: f32,
    _voxel_size_x: f32,
    _voxel_size_y: f32,
    _voxel_size_z: f32,
    _deskew_direction: i32,
    _auto_resize: bool,
) -> Result<ArrayPtr> {
    apply_affine_transform(src, dst, transform, false, false)
}

fn identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn translation(x: f32, y: f32, z: f32) -> Mat4 {
    let mut matrix = identity();
    matrix[0][3] = x;
    matrix[1][3] = y;
    matrix[2][3] = z;
    matrix
}

fn scale_matrix(x: f32, y: f32, z: f32) -> Mat4 {
    [
        [x, 0.0, 0.0, 0.0],
        [0.0, y, 0.0, 0.0],
        [0.0, 0.0, z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rotation_x(angle_deg: f32) -> Mat4 {
    let (sin, cos) = sin_cos_degrees(angle_deg);
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos, -sin, 0.0],
        [0.0, sin, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rotation_y(angle_deg: f32) -> Mat4 {
    let (sin, cos) = sin_cos_degrees(angle_deg);
    [
        [cos, 0.0, sin, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sin, 0.0, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rotation_z(angle_deg: f32) -> Mat4 {
    let (sin, cos) = sin_cos_degrees(angle_deg);
    [
        [cos, -sin, 0.0, 0.0],
        [sin, cos, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn shear_angle_to_shear_factor(angle_deg: f32) -> f32 {
    angle_deg.to_radians().tan()
}

fn sin_cos_degrees(angle_deg: f32) -> (f32, f32) {
    let (sin, cos) = angle_deg.to_radians().sin_cos();
    (zero_epsilon(sin), zero_epsilon(cos))
}

fn zero_epsilon(value: f32) -> f32 {
    if value.abs() < f32::EPSILON {
        0.0
    } else {
        value
    }
}

fn mul(left: Mat4, right: Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = left[row][0] * right[0][col]
                + left[row][1] * right[1][col]
                + left[row][2] * right[2][col]
                + left[row][3] * right[3][col];
        }
    }
    out
}

fn transpose(matrix: Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = matrix[col][row];
        }
    }
    out
}

fn invert_affine_matrix(matrix: Mat4) -> Mat4 {
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[0][2];
    let d = matrix[1][0];
    let e = matrix[1][1];
    let f = matrix[1][2];
    let g = matrix[2][0];
    let h = matrix[2][1];
    let i = matrix[2][2];

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv_det = 1.0 / det;

    let mut inv = identity();
    inv[0][0] = (e * i - f * h) * inv_det;
    inv[0][1] = (c * h - b * i) * inv_det;
    inv[0][2] = (b * f - c * e) * inv_det;
    inv[1][0] = (f * g - d * i) * inv_det;
    inv[1][1] = (a * i - c * g) * inv_det;
    inv[1][2] = (c * d - a * f) * inv_det;
    inv[2][0] = (d * h - e * g) * inv_det;
    inv[2][1] = (b * g - a * h) * inv_det;
    inv[2][2] = (a * e - b * d) * inv_det;

    let tx = matrix[0][3];
    let ty = matrix[1][3];
    let tz = matrix[2][3];
    inv[0][3] = -(inv[0][0] * tx + inv[0][1] * ty + inv[0][2] * tz);
    inv[1][3] = -(inv[1][0] * tx + inv[1][1] * ty + inv[1][2] * tz);
    inv[2][3] = -(inv[2][0] * tx + inv[2][1] * ty + inv[2][2] * tz);
    inv
}

fn flatten(matrix: Mat4) -> [f32; 16] {
    [
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[0][3],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[1][3],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
        matrix[2][3],
        matrix[3][0],
        matrix[3][1],
        matrix[3][2],
        matrix[3][3],
    ]
}
