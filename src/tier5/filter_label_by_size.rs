use crate::array::Array;
use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier3;
use crate::tier4;
use crate::types::{DType, MType, LABEL};

pub fn filter_label_by_size(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    minimum_size: f32,
    maximum_size: f32,
) -> Result<ArrayPtr> {
    let dst = tier0::create_like(src, dst, LABEL, device)?;
    let stats = tier3::statistics_of_background_and_labelled_pixels(device, None, Some(src))?;
    let nb_labels = stats["label"].len();
    let list_of_area = Array::create(nb_labels, 1, 1, 1, DType::Float, MType::Buffer, device)?;
    list_of_area
        .lock()
        .unwrap()
        .write_from(stats["area"].as_slice())?;
    tier4::remove_labels_with_map_values_out_of_range(
        device,
        src,
        &list_of_area,
        Some(dst),
        minimum_size,
        maximum_size,
    )
}

pub fn exclude_labels_outside_size_range(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
    minimum_size: f32,
    maximum_size: f32,
) -> Result<ArrayPtr> {
    filter_label_by_size(device, src, dst, minimum_size, maximum_size)
}
