use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

/// Crop `border_size` pixels from every side of the image.
pub fn crop_border(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    border_size: i32,
) -> Result<ArrayPtr> {
    let (width, height, depth) = {
        let src = src.lock().unwrap();
        (
            (src.width() as i32 - 2 * border_size).max(0) as usize,
            (src.height() as i32 - 2 * border_size).max(0) as usize,
            (src.depth() as i32 - 2 * border_size).max(0) as usize,
        )
    };
    tier1::crop(
        device,
        src,
        dst,
        border_size,
        border_size,
        border_size,
        width,
        height,
        depth,
    )
}
