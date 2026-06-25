use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::transform::{apply_affine_transform_deskew_3d, AffineTransform};

/// Deskews a volume as acquired with oblique plane light-sheet microscopy with
/// skew in the x direction.
pub fn deskew_x(
    _device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    angle: f32,
    voxel_size_x: f32,
    voxel_size_y: f32,
    voxel_size_z: f32,
    scale_factor: f32,
) -> Result<ArrayPtr> {
    let mut transform = AffineTransform::new();
    transform.deskew_x(
        angle,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        scale_factor,
    );
    apply_affine_transform_deskew_3d(
        src,
        dst,
        &transform,
        angle,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        0,
        true,
    )
}

/// Deskews a volume as acquired with oblique plane light-sheet microscopy with
/// skew in the y direction.
pub fn deskew_y(
    _device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    angle: f32,
    voxel_size_x: f32,
    voxel_size_y: f32,
    voxel_size_z: f32,
    scale_factor: f32,
) -> Result<ArrayPtr> {
    let mut transform = AffineTransform::new();
    transform.deskew_y(
        angle,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        scale_factor,
    );
    apply_affine_transform_deskew_3d(
        src,
        dst,
        &transform,
        angle,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        1,
        true,
    )
}
