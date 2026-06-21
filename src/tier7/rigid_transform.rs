use super::affine_transform::affine_transform;
use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;

type Mat4 = [[f32; 4]; 4];

/// Translates the image by a given vector and rotates it by given angles.
///
/// CLIc's C++ documentation currently says radians for this function, but the
/// implementation calls `AffineTransform::rotate`, which takes degrees. This
/// wrapper follows the implementation semantics: angles are given in degrees.
///
/// `centered` rotates around the image center when true, or the origin when
/// false. The current Rust affine helper does not implement CLIc's resize or
/// interpolation kernels yet, so `resize` and `interpolate` are accepted for API
/// parity but ignored.
pub fn rigid_transform(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    translate_x: f32,
    translate_y: f32,
    translate_z: f32,
    angle_x: f32,
    angle_y: f32,
    angle_z: f32,
    centered: bool,
    _interpolate: bool,
    _resize: bool,
) -> Result<ArrayPtr> {
    let (width, height, depth) = {
        let src = src.lock().unwrap();
        (src.width() as f32, src.height() as f32, src.depth() as f32)
    };

    let center = if centered {
        [
            if width != 1.0 { width / 2.0 } else { 0.0 },
            if height != 1.0 { height / 2.0 } else { 0.0 },
            if depth != 1.0 { depth / 2.0 } else { 0.0 },
        ]
    } else {
        [0.0, 0.0, 0.0]
    };

    let forward = compose_rigid_matrix(
        translate_x,
        translate_y,
        translate_z,
        angle_x,
        angle_y,
        angle_z,
        center,
    );
    let inv = invert_rigid_matrix(forward);
    affine_transform(device, src, dst, &flatten(inv))
}

fn compose_rigid_matrix(
    translate_x: f32,
    translate_y: f32,
    translate_z: f32,
    angle_x: f32,
    angle_y: f32,
    angle_z: f32,
    center: [f32; 3],
) -> Mat4 {
    let mut mat = translation(-center[0], -center[1], -center[2]);
    if angle_x != 0.0 {
        mat = mul(rotation_x(angle_x), mat);
    }
    if angle_y != 0.0 {
        mat = mul(rotation_y(angle_y), mat);
    }
    if angle_z != 0.0 {
        mat = mul(rotation_z(angle_z), mat);
    }
    mat = mul(translation(center[0], center[1], center[2]), mat);
    mul(translation(translate_x, translate_y, translate_z), mat)
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
    let mut mat = identity();
    mat[0][3] = x;
    mat[1][3] = y;
    mat[2][3] = z;
    mat
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

fn invert_rigid_matrix(mat: Mat4) -> Mat4 {
    let tx = mat[0][3];
    let ty = mat[1][3];
    let tz = mat[2][3];

    let mut inv = identity();
    for row in 0..3 {
        for col in 0..3 {
            inv[row][col] = mat[col][row];
        }
    }

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
