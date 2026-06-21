use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::Result;
use crate::tier1;

pub fn large_hessian_eigenvalue(
    device: &DeviceArc,
    src: &ArrayPtr,
    dst: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    let eigenvalues = tier1::hessian_eigenvalues(device, src)?;
    match dst {
        Some(dst) => tier1::copy(device, &eigenvalues[0], Some(dst)),
        None => Ok(eigenvalues[0].clone()),
    }
}
