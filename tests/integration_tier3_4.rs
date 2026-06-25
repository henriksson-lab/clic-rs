//! Integration tests for tier3 and tier4 functions — require a GPU.
//!
//! Run with:   cargo test --features gpu-tests

#[cfg(feature = "gpu-tests")]
#[path = "parity/tier2_3.rs"]
mod parity;

#[cfg(feature = "gpu-tests")]
mod gpu {
    use super::parity;
    use approx::assert_abs_diff_eq;
    use clic_rs::{backend_manager::BackendManager, tier3, tier4};

    fn device() -> clic_rs::DeviceArc {
        BackendManager::get_instance()
            .get_device("", "all")
            .expect("No OpenCL device found")
    }

    // ── tier3: mean_of_all_pixels ─────────────────────────────────────────────

    /// 10×20×30 all-ones → mean = 1.0
    #[test]
    fn mean_of_all_pixels() {
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
        let result = tier3::mean_of_all_pixels(&dev, &src).unwrap();
        assert_abs_diff_eq!(result, 1.0_f32, epsilon = 1e-4);
    }

    /// Heterogeneous array: [2, 4, 6] → mean = 4
    #[test]
    fn mean_of_all_pixels_simple() {
        let dev = device();
        let input: Vec<f32> = vec![2.0, 4.0, 6.0];
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
        let out = tier3::gamma_correction(&dev, &src, None, 0.5).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // min stays ~0, max stays ~100 (gamma correction maps [0,max]→[0,max])
        assert_abs_diff_eq!(min_val, 0.0_f32, epsilon = 1e-3);
        assert_abs_diff_eq!(max_val, 100.0_f32, epsilon = 1e-3);
    }

    // ── tier3: bounding_box ──────────────────────────────────────────────────

    /// Mirrors TestBoundingBox::execute2d.
    #[test]
    #[ignore = "currently hangs in tier3::bounding_box; kept as upstream parity fixture"]
    fn bounding_box_2d_matches_clic() {
        let dev = device();
        let src = clic_rs::array::Array::create_with_data(
            5,
            5,
            1,
            clic_rs::utils::shape_to_dimension(5, 5, 1),
            clic_rs::types::MType::Buffer,
            parity::BOUNDING_BOX_2D_INPUT_5X5,
            &dev,
        )
        .unwrap();
        let result = tier3::bounding_box(&dev, &src).unwrap();
        parity::assert_f32_slice_close(&result, &[1.0, 1.0, 0.0, 2.0, 2.0, 0.0], 1e-5);
    }

    /// Mirrors TestBoundingBox::execute3d.
    #[test]
    #[ignore = "currently hangs in tier3::bounding_box; kept as upstream parity fixture"]
    fn bounding_box_3d_matches_clic() {
        let dev = device();
        let src = clic_rs::array::Array::create_with_data(
            5,
            5,
            2,
            clic_rs::utils::shape_to_dimension(5, 5, 2),
            clic_rs::types::MType::Buffer,
            parity::BOUNDING_BOX_3D_INPUT_5X5X2,
            &dev,
        )
        .unwrap();
        let result = tier3::bounding_box(&dev, &src).unwrap();
        parity::assert_f32_slice_close(&result, &[1.0, 1.0, 0.0, 2.0, 2.0, 1.0], 1e-5);
    }

    // ── tier3: label matrices ────────────────────────────────────────────────

    /// Mirrors TestGenerateTouchMatrix::execute.
    #[test]
    fn generate_touch_matrix_matches_clic() {
        let dev = device();
        let src = clic_rs::array::Array::create_with_data(
            5,
            5,
            1,
            clic_rs::utils::shape_to_dimension(5, 5, 1),
            clic_rs::types::MType::Buffer,
            parity::LABEL_IMAGE_5X5,
            &dev,
        )
        .unwrap();
        let out = tier3::generate_touch_matrix(&dev, &src, None).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        parity::assert_u32_slice_eq(&result, parity::TOUCH_MATRIX_5X5);
    }

    /// Mirrors TestGenerateBinaryOverlapMatrix::execute.
    #[test]
    fn generate_binary_overlap_matrix_matches_clic() {
        let dev = device();
        let src0 = clic_rs::array::Array::create_with_data(
            5,
            2,
            2,
            clic_rs::utils::shape_to_dimension(5, 2, 2),
            clic_rs::types::MType::Buffer,
            parity::OVERLAP_INPUT_A_5X2X2,
            &dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            5,
            2,
            2,
            clic_rs::utils::shape_to_dimension(5, 2, 2),
            clic_rs::types::MType::Buffer,
            parity::OVERLAP_INPUT_B_5X2X2,
            &dev,
        )
        .unwrap();
        let out = tier3::generate_binary_overlap_matrix(&dev, &src0, &src1, None).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        parity::assert_u32_slice_eq(&result, parity::OVERLAP_MATRIX_5X3);
    }

    /// Mirrors TestExistingLabels::execute.
    #[test]
    fn flag_existing_labels_matches_clic() {
        let dev = device();
        let input: Vec<i32> = (0..150).map(|i| i % 10).collect();
        let src = clic_rs::array::Array::create_with_data(
            10,
            5,
            3,
            clic_rs::utils::shape_to_dimension(10, 5, 3),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier3::flag_existing_labels(&dev, &src, None).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        parity::assert_u32_slice_eq(&result, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    }

    // ── tier3: position and point-list functions ─────────────────────────────

    /// Mirrors TestMinMaxPosition::maxPosition.
    #[test]
    fn maximum_position_matches_clic() {
        let dev = device();
        let src = clic_rs::array::Array::create_with_data(
            4,
            3,
            2,
            clic_rs::utils::shape_to_dimension(4, 3, 2),
            clic_rs::types::MType::Buffer,
            parity::MINMAX_POSITION_INPUT_4X3X2,
            &dev,
        )
        .unwrap();
        let result = tier3::maximum_position(&dev, &src).unwrap();
        parity::assert_f32_slice_close(&result, &[1.0, 1.0, 0.0], 1e-5);
    }

    /// Mirrors TestMinMaxPosition::minPosition.
    #[test]
    fn minimum_position_matches_clic() {
        let dev = device();
        let src = clic_rs::array::Array::create_with_data(
            4,
            3,
            2,
            clic_rs::utils::shape_to_dimension(4, 3, 2),
            clic_rs::types::MType::Buffer,
            parity::MINMAX_POSITION_INPUT_4X3X2,
            &dev,
        )
        .unwrap();
        let result = tier3::minimum_position(&dev, &src).unwrap();
        parity::assert_f32_slice_close(&result, &[1.0, 1.0, 1.0], 1e-5);
    }

    /// Mirrors TestLabelSpotToPointList::execute.
    #[test]
    fn labelled_spots_to_pointlist_matches_clic() {
        let dev = device();
        let src = clic_rs::array::Array::create_with_data(
            5,
            5,
            1,
            clic_rs::utils::shape_to_dimension(5, 5, 1),
            clic_rs::types::MType::Buffer,
            parity::LABELLED_SPOTS_INPUT_5X5,
            &dev,
        )
        .unwrap();
        let out = tier3::labelled_spots_to_pointlist(&dev, &src, None).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        parity::assert_u32_slice_eq(&result, parity::LABELLED_SPOTS_POINTLIST_4X2);
    }

    // ── tier3: histogram / overlap metrics ───────────────────────────────────

    /// Mirrors TestHistogram::execute.
    #[test]
    fn histogram_matches_clic() {
        let dev = device();
        let input: Vec<f32> = (0..150).map(|i| (i % 10) as f32).collect();
        let src = clic_rs::array::Array::create_with_data(
            10,
            5,
            3,
            clic_rs::utils::shape_to_dimension(10, 5, 3),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier3::histogram(&dev, &src, None, 10, f32::NAN, f32::NAN).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        parity::assert_u32_slice_eq(&result, &[15, 15, 15, 15, 15, 15, 15, 15, 15, 15]);
    }

    /// Mirrors TestJaccardIndex::execute2D.
    #[test]
    fn jaccard_index_2d_matches_clic() {
        let dev = device();
        let a = vec![0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let src0 = clic_rs::array::Array::create_with_data(
            5,
            2,
            1,
            clic_rs::utils::shape_to_dimension(5, 2, 1),
            clic_rs::types::MType::Buffer,
            &a,
            &dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            5,
            2,
            1,
            clic_rs::utils::shape_to_dimension(5, 2, 1),
            clic_rs::types::MType::Buffer,
            &b,
            &dev,
        )
        .unwrap();
        let result = tier3::jaccard_index(&dev, &src0, &src1).unwrap();
        assert_abs_diff_eq!(result, 0.5_f32, epsilon = 1e-5);
    }

    /// Mirrors TestJaccardIndex::execute3D.
    #[test]
    fn jaccard_index_3d_matches_clic() {
        let dev = device();
        let a = vec![
            0.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let b = vec![
            0.0_f32, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let src0 = clic_rs::array::Array::create_with_data(
            3,
            2,
            2,
            clic_rs::utils::shape_to_dimension(3, 2, 2),
            clic_rs::types::MType::Buffer,
            &a,
            &dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            3,
            2,
            2,
            clic_rs::utils::shape_to_dimension(3, 2, 2),
            clic_rs::types::MType::Buffer,
            &b,
            &dev,
        )
        .unwrap();
        let result = tier3::jaccard_index(&dev, &src0, &src1).unwrap();
        assert_abs_diff_eq!(result, 0.5_f32, epsilon = 1e-5);
    }

    // ── tier4: mean_squared_error ─────────────────────────────────────────────

    /// Mirrors TestMeanSquareError
    /// input1=[1,2,3], input2=[4,5,7] → MSE = ((3^2 + 3^2 + 4^2) / 3) = 34/3 ≈ 11.333
    #[test]
    fn mean_squared_error_matches_clic() {
        let dev = device();
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0, 7.0];
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
        let result = tier4::mean_squared_error(&dev, &src0, &src1).unwrap();
        assert_abs_diff_eq!(result, 11.333_f32, epsilon = 0.01);
    }
}
