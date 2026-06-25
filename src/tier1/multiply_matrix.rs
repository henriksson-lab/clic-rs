use crate::array::ArrayPtr;
use crate::device::DeviceArc;
use crate::error::{CleError, Result};
use crate::execution::{execute, ConstantValue, ParameterValue};
use crate::tier0;
use crate::types::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MatmulConfig {
    tile_size: usize,
    wpt_m: usize,
    local_x: usize,
    local_y: usize,
    global_x: usize,
    global_y: usize,
}

fn next_power_of_2(mut v: usize) -> usize {
    if v == 0 {
        return 1;
    }
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if usize::BITS > 32 {
        v |= v >> 32;
    }
    v + 1
}

fn local_mem_required(tile_size: usize) -> usize {
    2 * tile_size * (tile_size + 1) * std::mem::size_of::<f32>()
}

fn validate_config(config: &MatmulConfig, device: &DeviceArc) -> bool {
    let wg_size = config.local_x * config.local_y;
    if wg_size > device.get_maximum_work_group_size() {
        return false;
    }
    if config.local_x > 256 || config.local_y > 256 {
        return false;
    }
    if local_mem_required(config.tile_size) > device.get_local_memory_size() {
        return false;
    }
    if config.tile_size % config.wpt_m != 0 {
        return false;
    }
    if config.local_y < 1 {
        return false;
    }
    true
}

fn select_wpt_m(tile_size: usize, device: &DeviceArc) -> usize {
    let max_wg_size = device.get_maximum_work_group_size();
    for wpt_m in [8_usize, 4, 2, 1] {
        if wpt_m > tile_size || tile_size % wpt_m != 0 {
            continue;
        }
        let local_y = tile_size / wpt_m;
        let wg_size = tile_size * local_y;
        if wg_size <= max_wg_size && local_y >= 1 {
            return wpt_m;
        }
    }
    1
}

fn select_tile_size(m: usize, k: usize, n: usize, device: &DeviceArc) -> usize {
    let min_dim = m.min(k).min(n);
    let ideal = ((min_dim as f64).sqrt() as usize).min(32);
    let ideal = next_power_of_2(ideal);
    for tile_size in [32_usize, 16, 8, 4, 2] {
        if tile_size > ideal {
            continue;
        }
        if local_mem_required(tile_size) > device.get_local_memory_size() {
            continue;
        }
        if tile_size * tile_size <= device.get_maximum_work_group_size() {
            return tile_size;
        }
    }
    1
}

fn build_fallback_config(m: usize, n: usize) -> MatmulConfig {
    MatmulConfig {
        tile_size: 1,
        wpt_m: 1,
        local_x: 1,
        local_y: 1,
        global_x: n,
        global_y: m,
    }
}

fn build_config(m: usize, k: usize, n: usize, device: &DeviceArc) -> MatmulConfig {
    if device.get_device_type() == "cpu" {
        return build_fallback_config(m, n);
    }

    if m <= 4 && k <= 4 && n <= 4 {
        return build_fallback_config(m, n);
    }

    let tile_size = select_tile_size(m, k, n, device);
    let wpt_m = select_wpt_m(tile_size, device);
    let ts = tile_size;
    let rts_m = ts / wpt_m;
    let mut config = MatmulConfig {
        tile_size,
        wpt_m,
        local_x: ts,
        local_y: rts_m,
        global_x: ((n + ts - 1) / ts) * ts,
        global_y: ((m + ts - 1) / ts) * ts / wpt_m,
    };

    if !validate_config(&config, device) {
        config.wpt_m = 1;
        config.local_y = ts;
        config.global_y = ((m + ts - 1) / ts) * ts;

        if !validate_config(&config, device) {
            config.tile_size = config.tile_size.min(4);
            let ts2 = config.tile_size;
            config.wpt_m = 1;
            config.local_x = ts2;
            config.local_y = ts2;
            config.global_x = ((n + ts2 - 1) / ts2) * ts2;
            config.global_y = ((m + ts2 - 1) / ts2) * ts2;

            if !validate_config(&config, device) {
                config = build_fallback_config(m, n);
            }
        }
    }

    config
}

/// Multiply two 2D matrices, interpreting `matrix1` as MxK and `matrix2` as KxN.
pub fn multiply_matrix(
    device: &DeviceArc,
    matrix1: &ArrayPtr,
    matrix2: &ArrayPtr,
    matrix_destination: Option<ArrayPtr>,
) -> Result<ArrayPtr> {
    {
        let matrix1 = matrix1.lock().unwrap();
        let matrix2 = matrix2.lock().unwrap();
        if matrix1.dim() > 2 || matrix2.dim() > 2 {
            eprintln!(
                "Warning: multiply_matrix expected 2D arrays but got {}D and {}D.",
                matrix1.dim(),
                matrix2.dim()
            );
        }
        if matrix1.width() != matrix2.height() {
            eprintln!(
                "Warning: matrix dimensions are not compatible for multiplication. Expected (M,K)x(K,N) but got ({},{})x({},{}).",
                matrix1.height(),
                matrix1.width(),
                matrix2.height(),
                matrix2.width()
            );
        }
    }

    let (dst_width, dst_height) = {
        let matrix1 = matrix1.lock().unwrap();
        let matrix2 = matrix2.lock().unwrap();
        (matrix2.width(), matrix1.height())
    };
    let dst = tier0::create_dst(
        matrix1,
        matrix_destination,
        dst_width,
        dst_height,
        1,
        DType::Float,
        device,
    )?;
    let (m, k, n) = {
        let matrix1 = matrix1.lock().unwrap();
        let dst = dst.lock().unwrap();
        (dst.height(), matrix1.width(), dst.width())
    };
    let config = build_config(m, k, n, device);

    let params = vec![
        ("src0", ParameterValue::Array(matrix1.clone())),
        ("src1", ParameterValue::Array(matrix2.clone())),
        ("dst", ParameterValue::Array(dst.clone())),
    ];
    let constants = vec![
        ("TILE_SIZE", ConstantValue::Int(config.tile_size as i32)),
        ("WPT_M", ConstantValue::Int(config.wpt_m as i32)),
    ];
    let kernel = (
        "multiply_matrix",
        include_str!("../../kernels/multiply_matrix.cl"),
    );
    let result = execute(
        device,
        kernel,
        &params,
        [config.global_x, config.global_y, 1],
        [config.local_x, config.local_y, 1],
        &constants,
    );
    if let Err(err) = result {
        if config.wpt_m > 1 {
            eprintln!(
                "Warning: multiply_matrix failed with TILE_SIZE={}, WPT_M={}. Retrying with WPT_M=1.\n  Error: {}",
                config.tile_size, config.wpt_m, err
            );
            let ts = config.tile_size;
            let fallback_constants = vec![
                ("TILE_SIZE", ConstantValue::Int(config.tile_size as i32)),
                ("WPT_M", ConstantValue::Int(1)),
            ];
            let fallback_result = execute(
                device,
                kernel,
                &params,
                [((n + ts - 1) / ts) * ts, ((m + ts - 1) / ts) * ts, 1],
                [ts, ts, 1],
                &fallback_constants,
            );
            match fallback_result {
                Ok(()) => return Ok(dst),
                Err(err2) => {
                    eprintln!(
                        "Warning: WPT_M=1 fallback also failed. Falling back to TILE_SIZE=1.\n  Error: {}",
                        err2
                    );
                }
            }
        } else if config.tile_size > 1 {
            eprintln!(
                "Warning: multiply_matrix failed with TILE_SIZE={}. Falling back to TILE_SIZE=1.\n  Error: {}",
                config.tile_size, err
            );
        }

        if config.tile_size > 1 {
            let safe = build_fallback_config(m, n);
            let safe_constants = vec![
                ("TILE_SIZE", ConstantValue::Int(1)),
                ("WPT_M", ConstantValue::Int(1)),
            ];
            execute(
                device,
                kernel,
                &params,
                [safe.global_x, safe.global_y, 1],
                [safe.local_x, safe.local_y, 1],
                &safe_constants,
            )
            .map_err(|err3| {
                CleError::Other(format!(
                    "multiply_matrix: all kernel configurations failed. Last error: {}",
                    err3
                ))
            })?;
        } else {
            return Err(CleError::Other(format!(
                "multiply_matrix: kernel execution failed with TILE_SIZE=1. Error: {}",
                err
            )));
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use opencl3::program::Program;
    use std::sync::Arc;

    struct FakeDevice {
        dtype: &'static str,
        max_wg: usize,
        local_mem: usize,
    }

    impl Device for FakeDevice {
        fn get_name(&self) -> &str {
            "fake"
        }
        fn get_device_type(&self) -> &str {
            self.dtype
        }
        fn support_image(&self) -> bool {
            false
        }
        fn get_maximum_buffer_size(&self) -> usize {
            usize::MAX
        }
        fn get_maximum_work_group_size(&self) -> usize {
            self.max_wg
        }
        fn get_local_memory_size(&self) -> usize {
            self.local_mem
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
            "fake".to_string()
        }
    }

    #[test]
    fn local_mem_required_matches_clic_padded_two_tile_formula() {
        assert_eq!(local_mem_required(1), 16);
        assert_eq!(local_mem_required(16), 2 * 16 * 17 * 4);
    }

    #[test]
    fn select_wpt_m_prefers_highest_valid_candidate() {
        let device: DeviceArc = Arc::new(FakeDevice {
            dtype: "gpu",
            max_wg: 256,
            local_mem: usize::MAX,
        });
        assert_eq!(select_wpt_m(32, &device), 8);

        let constrained: DeviceArc = Arc::new(FakeDevice {
            dtype: "gpu",
            max_wg: 16,
            local_mem: usize::MAX,
        });
        assert_eq!(select_wpt_m(32, &constrained), 1);
    }

    #[test]
    fn build_config_uses_cpu_and_tiny_fallbacks() {
        let cpu: DeviceArc = Arc::new(FakeDevice {
            dtype: "cpu",
            max_wg: 1024,
            local_mem: usize::MAX,
        });
        assert_eq!(
            build_config(128, 128, 128, &cpu),
            build_fallback_config(128, 128)
        );

        let gpu: DeviceArc = Arc::new(FakeDevice {
            dtype: "gpu",
            max_wg: 1024,
            local_mem: usize::MAX,
        });
        assert_eq!(build_config(4, 4, 4, &gpu), build_fallback_config(4, 4));
    }

    #[test]
    fn build_config_uses_wpt_m_geometry_for_gpu_path() {
        let device: DeviceArc = Arc::new(FakeDevice {
            dtype: "gpu",
            max_wg: 1024,
            local_mem: usize::MAX,
        });
        let config = build_config(1024, 1024, 1024, &device);

        assert_eq!(config.tile_size, 32);
        assert_eq!(config.wpt_m, 8);
        assert_eq!(config.local_x, 32);
        assert_eq!(config.local_y, 4);
        assert_eq!(config.global_x, 1024);
        assert_eq!(config.global_y, 128);
    }

    #[test]
    fn validate_config_rejects_local_memory_overflow() {
        let device: DeviceArc = Arc::new(FakeDevice {
            dtype: "gpu",
            max_wg: 1024,
            local_mem: local_mem_required(16) - 1,
        });
        let config = MatmulConfig {
            tile_size: 16,
            wpt_m: 1,
            local_x: 16,
            local_y: 16,
            global_x: 16,
            global_y: 16,
        };

        assert!(!validate_config(&config, &device));
    }
}
