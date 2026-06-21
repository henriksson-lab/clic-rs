use super::affine_transform::affine_transform;
use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;

type Mat4 = [[f32; 4]; 4];

/// Deskews a volume as acquired with oblique plane light-sheet microscopy with
/// skew in the x direction.
///
/// This is a compile-oriented wrapper using the generic affine helper. CLIc's
/// C++ implementation uses dedicated 3D deskew kernels with interpolation and
/// automatic output resizing; those kernels do not exist in this Rust tier yet.
/// The approximation below mirrors the affine shear/scale/rotation setup and
/// keeps the output shape chosen by the existing affine helper.
pub fn deskew_x(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    angle: f32,
    voxel_size_x: f32,
    _voxel_size_y: f32,
    voxel_size_z: f32,
    scale_factor: f32,
) -> Result<ArrayPtr> {
    let mut forward = identity();
    let angle_rad = angle.to_radians();
    let shear_factor = (90.0 - angle).to_radians().sin() * (voxel_size_z / voxel_size_x);
    forward[0][2] += shear_factor;

    let scale_factor_z = (angle_rad.sin() * voxel_size_z / voxel_size_x) * scale_factor;
    forward = mul(scale(scale_factor, scale_factor, scale_factor_z), forward);
    forward = mul(rotation_y(angle), forward);

    let inv = invert_affine_matrix(forward);
    affine_transform(device, src, dst, &flatten(inv))
}

/// Deskews a volume as acquired with oblique plane light-sheet microscopy with
/// skew in the y direction.
///
/// This is a compile-oriented wrapper using the generic affine helper. CLIc's
/// C++ implementation uses dedicated 3D deskew kernels with interpolation and
/// automatic output resizing; those kernels do not exist in this Rust tier yet.
/// The approximation below mirrors the affine shear/scale/rotation setup and
/// keeps the output shape chosen by the existing affine helper.
pub fn deskew_y(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    angle: f32,
    _voxel_size_x: f32,
    voxel_size_y: f32,
    voxel_size_z: f32,
    scale_factor: f32,
) -> Result<ArrayPtr> {
    let mut forward = identity();
    let angle_rad = angle.to_radians();
    let shear_factor = (90.0 - angle).to_radians().sin() * (voxel_size_z / voxel_size_y);
    forward[1][2] += shear_factor;

    let scale_factor_z = (angle_rad.sin() * voxel_size_z / voxel_size_y) * scale_factor;
    forward = mul(scale(scale_factor, scale_factor, scale_factor_z), forward);
    forward = mul(rotation_x(-angle), forward);

    let inv = invert_affine_matrix(forward);
    affine_transform(device, src, dst, &flatten(inv))
}

fn identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn scale(x: f32, y: f32, z: f32) -> Mat4 {
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

fn invert_affine_matrix(mat: Mat4) -> Mat4 {
    let a = mat[0][0];
    let b = mat[0][1];
    let c = mat[0][2];
    let d = mat[1][0];
    let e = mat[1][1];
    let f = mat[1][2];
    let g = mat[2][0];
    let h = mat[2][1];
    let i = mat[2][2];

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

    let tx = mat[0][3];
    let ty = mat[1][3];
    let tz = mat[2][3];
    inv[0][3] = -(inv[0][0] * tx + inv[0][1] * ty + inv[0][2] * tz);
    inv[1][3] = -(inv[1][0] * tx + inv[1][1] * ty + inv[1][2] * tz);
    inv[2][3] = -(inv[2][0] * tx + inv[2][1] * ty + inv[2][2] * tz);
    inv
}

fn flatten(mat: Mat4) -> [f32; 16] {
    [
        mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0], mat[1][1], mat[1][2], mat[1][3],
        mat[2][0], mat[2][1], mat[2][2], mat[2][3], mat[3][0], mat[3][1], mat[3][2], mat[3][3],
    ]
}
