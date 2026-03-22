//! Integration tests for tier3 and tier4 functions — require a GPU.
//!
//! Run with:   cargo test --features gpu-tests

#[cfg(feature = "gpu-tests")]
mod gpu {
    use approx::assert_abs_diff_eq;
    use clic_rs::{
        array::{pull, push},
        backend_manager::BackendManager,
        tier3, tier4,
    };

    fn device() -> clic_rs::DeviceArc {
        BackendManager::get()
            .get_device("", "all")
            .expect("No OpenCL device found")
    }

    // ── tier3: mean_of_all_pixels ─────────────────────────────────────────────

    /// 10×20×30 all-ones → mean = 1.0
    #[test]
    fn mean_of_all_pixels() {
        let dev = device();
        let input = vec![1.0_f32; 10 * 20 * 30];
        let src = push(&input, 10, 20, 30, &dev).unwrap();
        let result = tier3::mean_of_all_pixels(&dev, &src).unwrap();
        assert_abs_diff_eq!(result, 1.0_f32, epsilon = 1e-4);
    }

    /// Heterogeneous array: [2, 4, 6] → mean = 4
    #[test]
    fn mean_of_all_pixels_simple() {
        let dev = device();
        let input: Vec<f32> = vec![2.0, 4.0, 6.0];
        let src = push(&input, 3, 1, 1, &dev).unwrap();
        let result = tier3::mean_of_all_pixels(&dev, &src).unwrap();
        assert_abs_diff_eq!(result, 4.0_f32, epsilon = 1e-4);
    }

    // ── tier3: gamma_correction ───────────────────────────────────────────────

    /// Mirrors TestGammaCorrection — input has 0 and 100 as min/max.
    /// gamma=0.5: min stays ~0, max stays ~100.
    #[test]
    fn gamma_correction_preserves_range() {
        let dev = device();
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 50.0, 0.0, 5.0, 0.0,
            0.0, 0.0, 100.0, 0.0, 0.0,
            0.0, 30.0, 0.0, 10.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let src = push(&input, 5, 5, 1, &dev).unwrap();
        let out = tier3::gamma_correction(&dev, &src, None, 0.5).unwrap();
        let result: Vec<f32> = pull(&out).unwrap();
        let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // min stays ~0, max stays ~100 (gamma correction maps [0,max]→[0,max])
        assert_abs_diff_eq!(min_val, 0.0_f32, epsilon = 1e-3);
        assert_abs_diff_eq!(max_val, 100.0_f32, epsilon = 1e-3);
    }

    // ── tier4: mean_squared_error ─────────────────────────────────────────────

    /// Mirrors TestMeanSquareError
    /// input1=[1,2,3], input2=[4,5,7] → MSE = ((3^2 + 3^2 + 4^2) / 3) = 34/3 ≈ 11.333
    #[test]
    fn mean_squared_error_matches_clic() {
        let dev = device();
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0, 7.0];
        let src0 = push(&a, 3, 1, 1, &dev).unwrap();
        let src1 = push(&b, 3, 1, 1, &dev).unwrap();
        let result = tier4::mean_squared_error(&dev, &src0, &src1).unwrap();
        assert_abs_diff_eq!(result, 11.333_f32, epsilon = 0.01);
    }
}
