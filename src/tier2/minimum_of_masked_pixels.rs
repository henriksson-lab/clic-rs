use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier0;
use crate::tier1;
use crate::types::DType;

pub fn minimum_of_masked_pixels(
    device: &DeviceArc,
    src: &ArrayPtr,
    mask: &ArrayPtr,
) -> Result<f32> {
    let mut tmp_src = src.clone();
    let mut tmp_mask = mask.clone();

    if tmp_src.lock().unwrap().depth() > 1 {
        let dst_src = tier0::create_xy(src, None, DType::Unknown, device)?;
        let dst_mask = tier0::create_xy(mask, None, DType::Unknown, device)?;
        tier1::minimum_of_masked_pixels_reduction(
            device,
            &tmp_src,
            &tmp_mask,
            Some(dst_src.clone()),
            Some(dst_mask.clone()),
        )?;
        tmp_src = dst_src;
        tmp_mask = dst_mask;
    }

    if tmp_src.lock().unwrap().height() > 1 {
        tmp_src = tier1::transpose_yz(device, &tmp_src, None)?;
        tmp_mask = tier1::transpose_yz(device, &tmp_mask, None)?;
        let dst_src = tier0::create_vector(
            &tmp_src,
            None,
            tmp_src.lock().unwrap().width(),
            DType::Unknown,
            device,
        )?;
        let dst_mask = tier0::create_vector(
            &tmp_mask,
            None,
            tmp_mask.lock().unwrap().width(),
            DType::Unknown,
            device,
        )?;
        tier1::minimum_of_masked_pixels_reduction(
            device,
            &tmp_src,
            &tmp_mask,
            Some(dst_src.clone()),
            Some(dst_mask.clone()),
        )?;
        tmp_src = dst_src;
        tmp_mask = dst_mask;
    }

    tmp_src = tier1::transpose_xz(device, &tmp_src, None)?;
    tmp_mask = tier1::transpose_xz(device, &tmp_mask, None)?;
    let dst_src = tier0::create_one(&tmp_src, None, DType::Float, device)?;
    let dst_mask = tier0::create_one(&tmp_mask, None, DType::Float, device)?;
    tier1::minimum_of_masked_pixels_reduction(
        device,
        &tmp_src,
        &tmp_mask,
        Some(dst_src.clone()),
        Some(dst_mask),
    )?;

    let mut res = [0.0f32];
    dst_src.lock().unwrap().read_to(&mut res)?;
    Ok(res[0])
}
