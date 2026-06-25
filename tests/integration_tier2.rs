//! Integration tests for tier2 functions — require a GPU (OpenCL device).
//!
//! Run with:   cargo test --features gpu-tests
//!
//! Expected values are taken from CLIc's C++ test suite (`CLIc/tests/tier2/`).

#[cfg(feature = "gpu-tests")]
#[path = "parity/tier2_3.rs"]
mod parity;

#[cfg(feature = "gpu-tests")]
mod gpu {
    use super::parity;
    use approx::assert_abs_diff_eq;
    use clic_rs::{array::Array, backend_manager::BackendManager, tier2};

    fn device() -> clic_rs::DeviceArc {
        BackendManager::get_instance()
            .get_device("", "all")
            .expect("No OpenCL device found")
    }

    // ── absolute_difference ───────────────────────────────────────────────────

    /// Mirrors TestAbsoluteDifference::execute
    /// input1 = [1,5,3], input2 = [4,2,7] → valid = [3,3,4]
    #[test]
    fn absolute_difference_matches_clic() {
        let dev = device();
        let a: Vec<f32> = vec![1.0, 5.0, 3.0];
        let b: Vec<f32> = vec![4.0, 2.0, 7.0];
        let src0 = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &a,
            &dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &b,
            &dev,
        )
        .unwrap();
        let out = tier2::absolute_difference(&dev, &src0, &src1, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let valid = [3.0_f32, 3.0, 4.0];
        for (r, v) in result.iter().zip(valid.iter()) {
            assert_abs_diff_eq!(*r, *v, epsilon = 1e-5);
        }
    }

    // ── add_images / subtract_images ─────────────────────────────────────────

    /// Mirrors TestImagesOperation::add — input1=[1,5,3] + input2=[4,2,7] → [5,7,10]
    #[test]
    fn add_images_matches_clic() {
        let dev = device();
        let a: Vec<f32> = vec![1.0, 5.0, 3.0];
        let b: Vec<f32> = vec![4.0, 2.0, 7.0];
        let src0 = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &a,
            &dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &b,
            &dev,
        )
        .unwrap();
        let out = tier2::add_images(&dev, &src0, &src1, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let valid = [5.0_f32, 7.0, 10.0];
        for (r, v) in result.iter().zip(valid.iter()) {
            assert_abs_diff_eq!(*r, *v, epsilon = 1e-5);
        }
    }

    /// Mirrors TestImagesOperation::subtract — [1,5,3] - [4,2,7] → [-3,3,-4]
    #[test]
    fn subtract_images_matches_clic() {
        let dev = device();
        let a: Vec<f32> = vec![1.0, 5.0, 3.0];
        let b: Vec<f32> = vec![4.0, 2.0, 7.0];
        let src0 = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &a,
            &dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &b,
            &dev,
        )
        .unwrap();
        let out = tier2::subtract_images(&dev, &src0, &src1, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let valid = [-3.0_f32, 3.0, -4.0];
        for (r, v) in result.iter().zip(valid.iter()) {
            assert_abs_diff_eq!(*r, *v, epsilon = 1e-5);
        }
    }

    // ── square ────────────────────────────────────────────────────────────────

    /// Mirrors TestImagesOperation::square — [1,5,3] → [1,25,9]
    #[test]
    fn square_matches_clic() {
        let dev = device();
        let input: Vec<f32> = vec![1.0, 5.0, 3.0];
        let src = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier2::square(&dev, &src, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let valid = [1.0_f32, 25.0, 9.0];
        for (r, v) in result.iter().zip(valid.iter()) {
            assert_abs_diff_eq!(*r, *v, epsilon = 1e-4);
        }
    }

    // ── invert ────────────────────────────────────────────────────────────────

    /// Mirrors TestInvert — [1,-5,3] → [-1,5,-3]
    #[test]
    fn invert_matches_clic() {
        let dev = device();
        let input: Vec<f32> = vec![1.0, -5.0, 3.0];
        let src = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier2::invert(&dev, &src, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let valid = [-1.0_f32, 5.0, -3.0];
        for (r, v) in result.iter().zip(valid.iter()) {
            assert_abs_diff_eq!(*r, *v, epsilon = 1e-5);
        }
    }

    // ── clip ─────────────────────────────────────────────────────────────────

    /// Mirrors TestClip::executeMinMax — [0,1,2,3] clamp to [1,2] → [1,1,2,2]
    #[test]
    fn clip_matches_clic() {
        let dev = device();
        let input: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
        let src = clic_rs::array::Array::create_with_data(
            2,
            2,
            1,
            clic_rs::utils::shape_to_dimension(2, 2, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier2::clip(&dev, &src, None, 1.0, 2.0).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let valid = [1.0_f32, 1.0, 2.0, 2.0];
        for (r, v) in result.iter().zip(valid.iter()) {
            assert_abs_diff_eq!(*r, *v, epsilon = 1e-5);
        }
    }

    // ── degrees_to_radians ────────────────────────────────────────────────────

    /// Mirrors TestDegreeToRadiant — [180, 0, -90] → [π, 0, -π/2]
    #[test]
    fn degrees_to_radians_matches_clic() {
        let dev = device();
        let input: Vec<f32> = vec![180.0, 0.0, -90.0];
        let src = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier2::degrees_to_radians(&dev, &src, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        use std::f32::consts::PI;
        let valid = [PI, 0.0_f32, -0.5 * PI];
        for (r, v) in result.iter().zip(valid.iter()) {
            assert_abs_diff_eq!(*r, *v, epsilon = 1e-5);
        }
    }

    // ── maximum_of_all_pixels ─────────────────────────────────────────────────

    /// Mirrors TestMaxAllPixel — 10×20×30 array filled with 42, center = 100 → max = 100
    #[test]
    fn maximum_of_all_pixels_matches_clic() {
        let dev = device();
        let mut input = vec![42.0_f32; 10 * 20 * 30];
        let center = (10 / 2) + (20 / 2) * 10 + (30 / 2) * 10 * 20;
        input[center] = 100.0;
        let src = clic_rs::array::Array::create_with_data(
            10,
            20,
            30,
            clic_rs::utils::shape_to_dimension(10, 20, 30),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let result = tier2::maximum_of_all_pixels(&dev, &src).unwrap();
        assert_abs_diff_eq!(result, 100.0_f32, epsilon = 1e-4);
    }

    // ── minimum_of_all_pixels ─────────────────────────────────────────────────

    #[test]
    fn minimum_of_all_pixels_matches_clic() {
        let dev = device();
        let mut input = vec![42.0_f32; 10 * 20 * 30];
        let center = (10 / 2) + (20 / 2) * 10 + (30 / 2) * 10 * 20;
        input[center] = 100.0;
        let src = clic_rs::array::Array::create_with_data(
            10,
            20,
            30,
            clic_rs::utils::shape_to_dimension(10, 20, 30),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let result = tier2::minimum_of_all_pixels(&dev, &src).unwrap();
        assert_abs_diff_eq!(result, 42.0_f32, epsilon = 1e-4);
    }

    // ── sum_of_all_pixels ─────────────────────────────────────────────────────

    /// Mirrors TestSumAllPixel — 10×20×30 all-ones → sum = 6000
    #[test]
    fn sum_of_all_pixels_matches_clic() {
        let dev = device();
        let input = vec![1.0_f32; 10 * 20 * 30];
        let src = clic_rs::array::Array::create_with_data(
            10,
            20,
            30,
            clic_rs::utils::shape_to_dimension(10, 20, 30),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let result = tier2::sum_of_all_pixels(&dev, &src).unwrap();
        assert_abs_diff_eq!(result, 6000.0_f32, epsilon = 1.0);
    }

    // ── difference_of_gaussian ────────────────────────────────────────────────

    /// Mirrors TestDifferenceOfGaussian — 3×3×3 impulse at center, σ1=1, σ2=3
    #[test]
    fn difference_of_gaussian_matches_clic() {
        let dev = device();
        let mut input = vec![0.0_f32; 3 * 3 * 3];
        let center = (3 / 2) + (3 / 2) * 3 + (3 / 2) * 3 * 3;
        input[center] = 100.0;
        let src = clic_rs::array::Array::create_with_data(
            3,
            3,
            3,
            clic_rs::utils::shape_to_dimension(3, 3, 3),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier2::difference_of_gaussian(&dev, &src, None, [1.0, 1.0, 1.0], [3.0, 3.0, 3.0])
            .unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        // Expected from CLIc test (center value is largest)
        let valid = vec![
            1.217670321_f32,
            2.125371218,
            1.217670321,
            2.125371456,
            3.62864542,
            2.125371456,
            1.217670321,
            2.125371218,
            1.217670321,
            2.125371456,
            3.62864542,
            2.125371456,
            3.628645658,
            6.114237785,
            3.628645658,
            2.125371456,
            3.62864542,
            2.125371456,
            1.217670321,
            2.125371218,
            1.217670321,
            2.125371456,
            3.62864542,
            2.125371456,
            1.217670321,
            2.125371218,
            1.217670321,
        ];
        assert_eq!(result.len(), valid.len());
        for (r, v) in result.iter().zip(valid.iter()) {
            assert_abs_diff_eq!(*r, *v, epsilon = 1e-3);
        }
    }

    // ── closing_box ───────────────────────────────────────────────────────────

    /// Mirrors TestClosing::executeBox
    #[test]
    fn closing_box_matches_clic() {
        let dev = device();
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            0.0,0.0,0.0,0.0,0.0,0.0, 1.0,1.0,1.0,0.0,0.0,0.0, 1.0,1.0,1.0,0.0,2.0,0.0,
            1.0,1.0,1.0,0.0,2.0,0.0, 0.0,0.0,0.0,0.0,2.0,0.0, 3.0,0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0, 1.0,1.0,1.0,0.0,0.0,0.0, 1.0,1.0,1.0,0.0,2.0,0.0,
            1.0,1.0,1.0,0.0,2.0,0.0, 0.0,0.0,0.0,0.0,2.0,0.0, 3.0,0.0,0.0,0.0,0.0,0.0,
        ];
        #[rustfmt::skip]
        let valid: Vec<f32> = vec![
            1.0,1.0,1.0,0.0,0.0,0.0, 1.0,1.0,1.0,0.0,0.0,0.0, 1.0,1.0,1.0,1.0,2.0,2.0,
            1.0,1.0,1.0,1.0,2.0,2.0, 1.0,0.0,0.0,0.0,2.0,2.0, 3.0,0.0,0.0,0.0,2.0,2.0,
            1.0,1.0,1.0,0.0,0.0,0.0, 1.0,1.0,1.0,0.0,0.0,0.0, 1.0,1.0,1.0,1.0,2.0,2.0,
            1.0,1.0,1.0,1.0,2.0,2.0, 1.0,0.0,0.0,0.0,2.0,2.0, 3.0,0.0,0.0,0.0,2.0,2.0,
        ];
        let src = clic_rs::array::Array::create_with_data(
            6,
            6,
            2,
            clic_rs::utils::shape_to_dimension(6, 6, 2),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier2::closing_box(&dev, &src, None, 1.0, 1.0, 0.0).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        assert_eq!(result.len(), valid.len());
        for (r, v) in result.iter().zip(valid.iter()) {
            assert_abs_diff_eq!(*r, *v, epsilon = 1e-5);
        }
    }

    // ── opening_box ───────────────────────────────────────────────────────────

    /// Mirrors TestOpening::executeBox.
    #[test]
    fn opening_box_matches_clic() {
        let dev = device();
        let src = clic_rs::array::Array::create_with_data(
            6,
            6,
            2,
            clic_rs::utils::shape_to_dimension(6, 6, 2),
            clic_rs::types::MType::Buffer,
            parity::MORPHOLOGY_6X6X2,
            &dev,
        )
        .unwrap();
        let out = tier2::opening_box(&dev, &src, None, 1.0, 1.0, 0.0).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        parity::assert_f32_slice_close(&result, parity::OPENING_BOX_6X6X2, 1e-5);
    }

    /// Mirrors TestOpening::executeSphere.
    #[test]
    fn opening_sphere_matches_clic() {
        let dev = device();
        let src = clic_rs::array::Array::create_with_data(
            6,
            6,
            2,
            clic_rs::utils::shape_to_dimension(6, 6, 2),
            clic_rs::types::MType::Buffer,
            parity::MORPHOLOGY_6X6X2,
            &dev,
        )
        .unwrap();
        let out = tier2::opening_sphere(&dev, &src, None, 1.0, 1.0, 0.0).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        parity::assert_f32_slice_close(&result, parity::OPENING_SPHERE_6X6X2, 1e-5);
    }

    // ── top_hat_box ───────────────────────────────────────────────────────────

    /// Mirrors TestTopHat::executeBox — input has 100 at center, 50 around it.
    /// Top-hat result min=0, max=50
    #[test]
    fn top_hat_box_matches_clic() {
        let dev = device();
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            0.0,0.0,0.0,0.0,0.0, 0.0,50.0,50.0,50.0,0.0,
            0.0,50.0,100.0,50.0,0.0, 0.0,50.0,50.0,50.0,0.0,
            0.0,0.0,0.0,0.0,0.0,
        ];
        let src = clic_rs::array::Array::create_with_data(
            5,
            5,
            1,
            clic_rs::utils::shape_to_dimension(5, 5, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier2::top_hat_box(&dev, &src, None, 1.0, 1.0, 0.0).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_abs_diff_eq!(min_val, 0.0_f32, epsilon = 1e-4);
        assert_abs_diff_eq!(max_val, 50.0_f32, epsilon = 1e-4);
    }

    // ── top_hat_sphere ────────────────────────────────────────────────────────

    /// Mirrors TestTopHat::executeSphere — same input, min=0, max=50
    #[test]
    fn top_hat_sphere_matches_clic() {
        let dev = device();
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            0.0,0.0,0.0,0.0,0.0, 0.0,50.0,50.0,50.0,0.0,
            0.0,50.0,100.0,50.0,0.0, 0.0,50.0,50.0,50.0,0.0,
            0.0,0.0,0.0,0.0,0.0,
        ];
        let src = clic_rs::array::Array::create_with_data(
            5,
            5,
            1,
            clic_rs::utils::shape_to_dimension(5, 5, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier2::top_hat_sphere(&dev, &src, None, 1.0, 1.0, 0.0).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_abs_diff_eq!(min_val, 0.0_f32, epsilon = 1e-4);
        assert_abs_diff_eq!(max_val, 50.0_f32, epsilon = 1e-4);
    }

    // ── concatenate ──────────────────────────────────────────────────────────

    /// Mirrors TestConcatenate::alongX.
    #[test]
    fn concatenate_along_x_matches_clic() {
        let dev = device();
        let a = vec![1.0_f32; 4];
        let b = vec![2.0_f32; 4];
        let src0 = clic_rs::array::Array::create_with_data(
            2,
            2,
            1,
            clic_rs::utils::shape_to_dimension(2, 2, 1),
            clic_rs::types::MType::Buffer,
            &a,
            &dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            2,
            2,
            1,
            clic_rs::utils::shape_to_dimension(2, 2, 1),
            clic_rs::types::MType::Buffer,
            &b,
            &dev,
        )
        .unwrap();
        let out = tier2::concatenate_along_x(&dev, &src0, &src1, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        parity::assert_f32_slice_close(&result, &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0], 1e-5);
        let out = out.lock().unwrap();
        assert_eq!((out.width(), out.height(), out.depth()), (4, 2, 1));
    }

    /// Mirrors TestConcatenate::alongY.
    #[test]
    fn concatenate_along_y_matches_clic() {
        let dev = device();
        let a = vec![1.0_f32; 4];
        let b = vec![2.0_f32; 4];
        let src0 = clic_rs::array::Array::create_with_data(
            2,
            2,
            1,
            clic_rs::utils::shape_to_dimension(2, 2, 1),
            clic_rs::types::MType::Buffer,
            &a,
            &dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            2,
            2,
            1,
            clic_rs::utils::shape_to_dimension(2, 2, 1),
            clic_rs::types::MType::Buffer,
            &b,
            &dev,
        )
        .unwrap();
        let out = tier2::concatenate_along_y(&dev, &src0, &src1, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        parity::assert_f32_slice_close(&result, &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], 1e-5);
        let out = out.lock().unwrap();
        assert_eq!((out.width(), out.height(), out.depth()), (2, 4, 1));
    }

    /// Mirrors TestConcatenate::alongZ.
    #[test]
    fn concatenate_along_z_matches_clic() {
        let dev = device();
        let a = vec![1.0_f32; 4];
        let b = vec![2.0_f32; 4];
        let src0 = clic_rs::array::Array::create_with_data(
            2,
            2,
            1,
            clic_rs::utils::shape_to_dimension(2, 2, 1),
            clic_rs::types::MType::Buffer,
            &a,
            &dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            2,
            2,
            1,
            clic_rs::utils::shape_to_dimension(2, 2, 1),
            clic_rs::types::MType::Buffer,
            &b,
            &dev,
        )
        .unwrap();
        let out = tier2::concatenate_along_z(&dev, &src0, &src1, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        parity::assert_f32_slice_close(&result, &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], 1e-5);
        let out = out.lock().unwrap();
        assert_eq!((out.width(), out.height(), out.depth()), (2, 2, 2));
    }

    // ── pointlist_to_labelled_spots ────────────────────────────────────────

    /// Mirrors CLIc's `pointlist_to_labelled_spots_func` for a small 2D point-list.
    #[test]
    fn pointlist_to_labelled_spots_2d() {
        let dev = device();
        let input: Vec<f32> = vec![1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0];
        let pointlist = clic_rs::array::Array::create_with_data(
            3,
            3,
            1,
            clic_rs::utils::shape_to_dimension(3, 3, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier2::pointlist_to_labelled_spots(&dev, &pointlist, None).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let valid = vec![0_u32, 1, 0, 0, 0, 3, 2, 0, 0];
        assert_eq!(result, valid);
    }

    /// Uses a user-provided destination buffer with larger shape.
    #[test]
    fn pointlist_to_labelled_spots_with_destination() {
        let dev = device();
        let input: Vec<f32> = vec![1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0];
        let pointlist = clic_rs::array::Array::create_with_data(
            3,
            3,
            1,
            clic_rs::utils::shape_to_dimension(3, 3, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let dst = Array::create(
            5,
            5,
            1,
            2,
            clic_rs::types::LABEL,
            clic_rs::types::MType::Buffer,
            &dev,
        )
        .unwrap();
        let out = tier2::pointlist_to_labelled_spots(&dev, &pointlist, Some(dst)).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let valid = vec![
            0_u32, 1, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(result, valid);
    }
}
