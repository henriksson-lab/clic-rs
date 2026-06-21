use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

pub fn small_hessian_eigenvalue(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let eigenvalues = tier1::hessian_eigenvalues(device, src)?;
    let small = eigenvalues
        .last()
        .expect("hessian_eigenvalues returns at least large and small eigenvalues");
    match dst {
        Some(dst) => tier1::copy(device, small, Some(dst)),
        None => Ok(small.clone()),
    }
}
