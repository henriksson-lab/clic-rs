//! Integration tests for tier5 and tier7 functions — require a GPU.
//!
//! Run with:   cargo test --features gpu-tests

#[cfg(feature = "gpu-tests")]
mod gpu {
    use approx::assert_abs_diff_eq;
    use clic_rs::{
        array::{pull, push},
        backend_manager::BackendManager,
        tier5, tier7,
    };

    fn device() -> clic_rs::DeviceArc {
        BackendManager::get()
            .get_device("", "all")
            .expect("No OpenCL device found")
    }

    // ── tier5: array_equal ────────────────────────────────────────────────────

    /// Identical arrays → equal.
    #[test]
    fn array_equal_identical() {
        let dev = device();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let a = push(&data, 4, 1, 1, &dev).unwrap();
        let b = push(&data, 4, 1, 1, &dev).unwrap();
        assert!(tier5::array_equal(&dev, &a, &b).unwrap());
    }

    /// Different values → not equal.
    #[test]
    fn array_equal_different_values() {
        let dev = device();
        let a = push(&vec![1.0_f32, 2.0, 3.0], 3, 1, 1, &dev).unwrap();
        let b = push(&vec![1.0_f32, 2.0, 4.0], 3, 1, 1, &dev).unwrap();
        assert!(!tier5::array_equal(&dev, &a, &b).unwrap());
    }

    /// Different shapes → not equal (no GPU work needed).
    #[test]
    fn array_equal_different_shapes() {
        let dev = device();
        let a = push(&vec![1.0_f32; 4], 4, 1, 1, &dev).unwrap();
        let b = push(&vec![1.0_f32; 4], 2, 2, 1, &dev).unwrap();
        assert!(!tier5::array_equal(&dev, &a, &b).unwrap());
    }

    // ── tier7: translate ──────────────────────────────────────────────────────

    /// translate_x=2: spike at pixel 1 → appears at pixel 3.
    /// Input:  [0, 10, 0, 0, 0]
    /// Output: [0, 0, 0, 10, 0]  (content shifts right by 2)
    #[test]
    fn translate_x_spike() {
        let dev = device();
        let input: Vec<f32> = vec![0.0, 10.0, 0.0, 0.0, 0.0];
        let src = push(&input, 5, 1, 1, &dev).unwrap();
        let out = tier7::translate(&dev, &src, None, 2.0, 0.0, 0.0).unwrap();
        let result: Vec<f32> = pull(&out).unwrap();
        // dst[3] = src[1] = 10
        assert_abs_diff_eq!(result[3], 10.0_f32, epsilon = 1e-3);
        assert_abs_diff_eq!(result[0], 0.0_f32, epsilon = 1e-3);
    }

    /// translate_x=-1: spike at pixel 2 → appears at pixel 1.
    #[test]
    fn translate_x_negative() {
        let dev = device();
        let input: Vec<f32> = vec![0.0, 0.0, 7.0, 0.0, 0.0];
        let src = push(&input, 5, 1, 1, &dev).unwrap();
        let out = tier7::translate(&dev, &src, None, -1.0, 0.0, 0.0).unwrap();
        let result: Vec<f32> = pull(&out).unwrap();
        // dst[1] = src[2] = 7
        assert_abs_diff_eq!(result[1], 7.0_f32, epsilon = 1e-3);
    }

    /// translate with identity (tx=ty=tz=0) → array unchanged.
    #[test]
    fn translate_identity() {
        let dev = device();
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let src = push(&input, 5, 1, 1, &dev).unwrap();
        let out = tier7::translate(&dev, &src, None, 0.0, 0.0, 0.0).unwrap();
        let result: Vec<f32> = pull(&out).unwrap();
        for (a, b) in input.iter().zip(result.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-3);
        }
    }

    // ── tier7: scale ──────────────────────────────────────────────────────────

    /// scale_x=2: a [1, 2, 3, 4] array scaled by 2 maps dst[i] = src[i/2].
    /// With 8 output pixels: [1,1,2,2,3,3,4,4] approximately.
    #[test]
    fn scale_x_stretches() {
        let dev = device();
        // Uniform array: scaling shouldn't change values
        let input = vec![5.0_f32; 4];
        let src = push(&input, 4, 1, 1, &dev).unwrap();
        let out = tier7::scale(&dev, &src, None, 2.0, 1.0, 1.0).unwrap();
        let result: Vec<f32> = pull(&out).unwrap();
        // dst[0] = src[0/2.0 = 0] = 5, dst[1] = src[1/2.0 = 0] = 5, etc.
        // all pixels should be 5 (uniform source)
        for v in result.iter().take(2) {
            assert_abs_diff_eq!(*v, 5.0_f32, epsilon = 1e-3);
        }
    }

    /// scale identity (sx=sy=sz=1) → array unchanged.
    #[test]
    fn scale_identity() {
        let dev = device();
        let input: Vec<f32> = vec![1.0, 2.0, 3.0];
        let src = push(&input, 3, 1, 1, &dev).unwrap();
        let out = tier7::scale(&dev, &src, None, 1.0, 1.0, 1.0).unwrap();
        let result: Vec<f32> = pull(&out).unwrap();
        for (a, b) in input.iter().zip(result.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-3);
        }
    }
}
