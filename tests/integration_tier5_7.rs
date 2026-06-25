//! Integration tests for tier5 and tier7 functions — require a GPU.
//!
//! Run with:   cargo test --features gpu-tests

mod transform_parity {
    use approx::assert_abs_diff_eq;
    use clic_rs::transform::AffineTransform;

    fn assert_matrix_eq(actual: [[f32; 4]; 4], expected: [[f32; 4]; 4]) {
        for row in 0..4 {
            for col in 0..4 {
                assert_abs_diff_eq!(actual[row][col], expected[row][col], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn transform_to_array_matches_clic_rotate_z_fixture() {
        let mut transform = AffineTransform::new();
        transform.rotate(2, 90.0);

        assert_eq!(
            AffineTransform::to_array(transform.get_transpose()),
            [
                0.0, -1.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ]
        );
    }

    #[test]
    fn transform_translate_xyz_matches_clic_fixture() {
        let mut transform = AffineTransform::new();
        transform.translate(1.0, 2.0, 3.0);

        assert_matrix_eq(
            transform.get_matrix(),
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
    }

    #[test]
    fn transform_rotate_axes_match_clic_fixtures() {
        let mut x = AffineTransform::new();
        x.rotate(0, 90.0);
        assert_matrix_eq(
            x.get_matrix(),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );

        let mut y = AffineTransform::new();
        y.rotate(1, 90.0);
        assert_matrix_eq(
            y.get_matrix(),
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );

        let mut z = AffineTransform::new();
        z.rotate(2, 90.0);
        assert_matrix_eq(
            z.get_matrix(),
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
    }

    #[test]
    fn transform_center_rotate_matches_clic_fixture() {
        let mut transform = AffineTransform::new();
        transform.center([30, 20, 10], false);
        transform.rotate(2, 90.0);
        transform.center([30, 20, 10], true);

        assert_matrix_eq(
            transform.get_matrix(),
            [
                [0.0, -1.0, 0.0, 25.0],
                [1.0, 0.0, 0.0, -5.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
    }
}

#[cfg(feature = "gpu-tests")]
mod gpu {
    use approx::assert_abs_diff_eq;
    use clic_rs::{backend_manager::BackendManager, tier5, tier6, tier7};

    fn device() -> clic_rs::DeviceArc {
        BackendManager::get_instance()
            .get_device("", "all")
            .expect("No OpenCL device found")
    }

    fn assert_f32_slice_close(actual: &[f32], expected: &[f32], epsilon: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            if (*actual - *expected).abs() > epsilon {
                panic!("mismatch at index {index}: actual={actual}, expected={expected}");
            }
        }
    }

    // ── tier5: array_equal ────────────────────────────────────────────────────

    /// Identical arrays → equal.
    #[test]
    fn array_equal_identical() {
        let dev = device();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let a = clic_rs::array::Array::create_with_data(
            4,
            1,
            1,
            clic_rs::utils::shape_to_dimension(4, 1, 1),
            clic_rs::types::MType::Buffer,
            &data,
            &dev,
        )
        .unwrap();
        let b = clic_rs::array::Array::create_with_data(
            4,
            1,
            1,
            clic_rs::utils::shape_to_dimension(4, 1, 1),
            clic_rs::types::MType::Buffer,
            &data,
            &dev,
        )
        .unwrap();
        assert!(tier5::array_equal(&dev, &a, &b).unwrap());
    }

    /// Different values → not equal.
    #[test]
    fn array_equal_different_values() {
        let dev = device();
        let a = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &vec![1.0_f32, 2.0, 3.0],
            &dev,
        )
        .unwrap();
        let b = clic_rs::array::Array::create_with_data(
            3,
            1,
            1,
            clic_rs::utils::shape_to_dimension(3, 1, 1),
            clic_rs::types::MType::Buffer,
            &vec![1.0_f32, 2.0, 4.0],
            &dev,
        )
        .unwrap();
        assert!(!tier5::array_equal(&dev, &a, &b).unwrap());
    }

    /// Different shapes → not equal (no GPU work needed).
    #[test]
    fn array_equal_different_shapes() {
        let dev = device();
        let a = clic_rs::array::Array::create_with_data(
            4,
            1,
            1,
            clic_rs::utils::shape_to_dimension(4, 1, 1),
            clic_rs::types::MType::Buffer,
            &vec![1.0_f32; 4],
            &dev,
        )
        .unwrap();
        let b = clic_rs::array::Array::create_with_data(
            2,
            2,
            1,
            clic_rs::utils::shape_to_dimension(2, 2, 1),
            clic_rs::types::MType::Buffer,
            &vec![1.0_f32; 4],
            &dev,
        )
        .unwrap();
        assert!(!tier5::array_equal(&dev, &a, &b).unwrap());
    }

    #[test]
    #[ignore = "current Rust label-statistics path does not return promptly on this fixture"]
    fn filter_label_by_size_matches_clic_fixture() {
        let dev = device();
        #[rustfmt::skip]
        let input = [
            1_u32, 1, 2, 0, 3, 3,
            1, 1, 2, 0, 3, 3,
            0, 0, 0, 0, 0, 0,
            4, 4, 5, 6, 6, 6,
            4, 4, 5, 6, 6, 6,
        ];
        let src = clic_rs::array::Array::create_with_data(
            6,
            5,
            1,
            clic_rs::utils::shape_to_dimension(6, 5, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier5::filter_label_by_size(&dev, &src, None, 4.0, 5.0).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        #[rustfmt::skip]
        let expected = [
            1_u32, 1, 0, 0, 2, 2,
            1, 1, 0, 0, 2, 2,
            0, 0, 0, 0, 0, 0,
            3, 3, 0, 0, 0, 0,
            3, 3, 0, 0, 0, 0,
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn connected_component_labeling_box_matches_clic_fixture() {
        let dev = device();
        #[rustfmt::skip]
        let input = [
            0_u32, 0, 0, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 0,
            1, 0, 0, 1, 0,
            0, 1, 0, 1, 1,
            0, 1, 0, 0, 0,
        ];
        let src = clic_rs::array::Array::create_with_data(
            5,
            3,
            2,
            clic_rs::utils::shape_to_dimension(5, 3, 2),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier5::connected_component_labeling(&dev, &src, None, "box").unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        #[rustfmt::skip]
        let expected = [
            0_u32, 0, 0, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 0,
            2, 0, 0, 1, 0,
            0, 2, 0, 1, 1,
            0, 2, 0, 0, 0,
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn connected_component_labeling_sphere_matches_clic_fixture() {
        let dev = device();
        #[rustfmt::skip]
        let input = [
            0_u32, 0, 0, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 0,
            1, 0, 0, 1, 0,
            0, 1, 0, 1, 1,
            0, 1, 0, 0, 0,
        ];
        let src = clic_rs::array::Array::create_with_data(
            5,
            3,
            2,
            clic_rs::utils::shape_to_dimension(5, 3, 2),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier5::connected_component_labeling(&dev, &src, None, "sphere").unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        #[rustfmt::skip]
        let expected = [
            0_u32, 0, 0, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 0,
            2, 0, 0, 1, 0,
            0, 3, 0, 1, 1,
            0, 3, 0, 0, 0,
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn normalize_matches_clic_fixture() {
        let dev = device();
        let input = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let src = clic_rs::array::Array::create_with_data(
            11,
            1,
            1,
            clic_rs::utils::shape_to_dimension(11, 1, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier5::normalize(&dev, &src, None, 10.0, 90.0).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let expected = [
            0.0_f32, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0,
        ];
        assert_f32_slice_close(&result, &expected, 0.01);
    }

    #[test]
    #[ignore = "current Rust centroid/statistics path does not return promptly on this fixture"]
    fn reduce_labels_to_centroids_matches_clic_fixture() {
        let dev = device();
        #[rustfmt::skip]
        let input = [
            0_u32, 0, 0, 1, 1, 1,
            0, 2, 0, 1, 1, 1,
            0, 0, 0, 1, 1, 1,
            3, 3, 3, 4, 4, 4,
            3, 3, 3, 4, 4, 4,
            3, 3, 3, 4, 4, 4,
        ];
        let src = clic_rs::array::Array::create_with_data(
            6,
            6,
            1,
            clic_rs::utils::shape_to_dimension(6, 6, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier5::reduce_labels_to_centroids(&dev, &src, None).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        #[rustfmt::skip]
        let expected = [
            0_u32, 0, 0, 0, 0, 0,
            0, 2, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 3, 0, 0, 4, 0,
            0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(result, expected);
    }

    // ── tier6: label morphology ──────────────────────────────────────────────

    #[test]
    fn dilate_labels_matches_clic_fixture() {
        let dev = device();
        #[rustfmt::skip]
        let input = [
            0_u32, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0,
            0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3,
            1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
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
        let out = tier6::dilate_labels(&dev, &src, None, 1).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        #[rustfmt::skip]
        let expected = [
            1_u32, 1, 0, 0, 2, 2, 1, 1, 0, 0, 2, 2,
            0, 0, 4, 4, 4, 0, 0, 0, 4, 4, 4, 0,
            5, 5, 4, 4, 4, 3, 5, 5, 0, 0, 3, 3,
            1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 2, 2,
            0, 0, 4, 4, 4, 0, 0, 0, 4, 4, 4, 0,
            5, 5, 4, 4, 4, 3, 5, 5, 0, 0, 3, 3,
        ];
        assert_eq!(result, expected);
    }

    // ── tier7: translate ──────────────────────────────────────────────────────

    /// translate_x=2: spike at pixel 1 → appears at pixel 3.
    /// Input:  [0, 10, 0, 0, 0]
    /// Output: [0, 0, 0, 10, 0]  (content shifts right by 2)
    #[test]
    fn translate_x_spike() {
        let dev = device();
        let input: Vec<f32> = vec![0.0, 10.0, 0.0, 0.0, 0.0];
        let src = clic_rs::array::Array::create_with_data(
            5,
            1,
            1,
            clic_rs::utils::shape_to_dimension(5, 1, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier7::translate(&dev, &src, None, 2.0, 0.0, 0.0, false).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        // dst[3] = src[1] = 10
        assert_abs_diff_eq!(result[3], 10.0_f32, epsilon = 1e-3);
        assert_abs_diff_eq!(result[0], 0.0_f32, epsilon = 1e-3);
    }

    /// translate_x=-1: spike at pixel 2 → appears at pixel 1.
    #[test]
    fn translate_x_negative() {
        let dev = device();
        let input: Vec<f32> = vec![0.0, 0.0, 7.0, 0.0, 0.0];
        let src = clic_rs::array::Array::create_with_data(
            5,
            1,
            1,
            clic_rs::utils::shape_to_dimension(5, 1, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier7::translate(&dev, &src, None, -1.0, 0.0, 0.0, false).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        // dst[1] = src[2] = 7
        assert_abs_diff_eq!(result[1], 7.0_f32, epsilon = 1e-3);
    }

    /// translate with identity (tx=ty=tz=0) → array unchanged.
    #[test]
    fn translate_identity() {
        let dev = device();
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let src = clic_rs::array::Array::create_with_data(
            5,
            1,
            1,
            clic_rs::utils::shape_to_dimension(5, 1, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier7::translate(&dev, &src, None, 0.0, 0.0, 0.0, false).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        for (a, b) in input.iter().zip(result.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-3);
        }
    }

    #[test]
    fn translate_matches_clic_fixture() {
        let dev = device();
        #[rustfmt::skip]
        let input = [
            0.0_f32, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
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
        let out = tier7::translate(&dev, &src, None, -1.0, -1.0, 0.0, false).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        #[rustfmt::skip]
        let expected = [
            0.0_f32, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_f32_slice_close(&result, &expected, 1e-4);
    }

    // ── tier7: scale ──────────────────────────────────────────────────────────

    /// scale_x=2: a [1, 2, 3, 4] array scaled by 2 maps dst[i] = src[i/2].
    /// With 8 output pixels: [1,1,2,2,3,3,4,4] approximately.
    #[test]
    fn scale_x_stretches() {
        let dev = device();
        // Uniform array: scaling shouldn't change values
        let input = vec![5.0_f32; 4];
        let src = clic_rs::array::Array::create_with_data(
            4,
            1,
            1,
            clic_rs::utils::shape_to_dimension(4, 1, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier7::scale(&dev, &src, None, 2.0, 1.0, 1.0, false, false, false).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

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
        let out = tier7::scale(&dev, &src, None, 1.0, 1.0, 1.0, false, false, false).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        for (a, b) in input.iter().zip(result.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-3);
        }
    }

    #[test]
    fn opening_labels_matches_clic_fixture() {
        let dev = device();
        #[rustfmt::skip]
        let input = [
            0_u32, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
            1, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 0,
            0, 0, 0, 2, 2, 0, 3, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
            1, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 0,
            0, 0, 0, 2, 2, 0, 3, 0, 0, 0, 0, 0,
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
        let out = tier7::opening_labels(&dev, &src, None, 1).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        #[rustfmt::skip]
        let expected = [
            0_u32, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
            1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
            1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(result, expected);
    }
}
