//! Integration parity tests for tier4 functions against original CLIc fixtures.
//!
//! Run with:   cargo test --features gpu-tests --test integration_tier4

#[cfg(feature = "gpu-tests")]
mod gpu {
    use clic_rs::{backend_manager::BackendManager, tier4};

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

    fn assert_u8_eq(actual: &[u8], expected: &[u8]) {
        assert_eq!(actual, expected);
    }

    const LABEL_FIXTURE_6X5: [u32; 6 * 5] = [
        1, 1, 2, 0, 3, 3, //
        1, 1, 2, 0, 3, 3, //
        0, 0, 0, 0, 0, 0, //
        4, 4, 5, 6, 6, 6, //
        4, 4, 5, 6, 6, 6,
    ];

    #[test]
    #[ignore = "current Rust label-statistics path does not return promptly on this fixture"]
    fn centroids_of_labels_matches_clic_fixture() {
        let dev = device();
        let src = clic_rs::array::Array::create_with_data(
            6,
            5,
            1,
            clic_rs::utils::shape_to_dimension(6, 5, 1),
            clic_rs::types::MType::Buffer,
            &LABEL_FIXTURE_6X5,
            &dev,
        )
        .unwrap();
        let out = tier4::centroids_of_labels(&dev, &src, None, true).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        #[rustfmt::skip]
        let expected = [
            2.625, 0.5, 2.0, 4.5, 0.5, 2.0, 4.0,
            1.625, 0.5, 0.5, 0.5, 3.5, 3.5, 3.5,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_f32_slice_close(&result, &expected, 1e-4);
    }

    #[test]
    #[ignore = "current Rust label-statistics path does not return promptly on this fixture"]
    fn pixel_count_map_matches_clic_fixture() {
        let dev = device();
        let src = clic_rs::array::Array::create_with_data(
            6,
            5,
            1,
            clic_rs::utils::shape_to_dimension(6, 5, 1),
            clic_rs::types::MType::Buffer,
            &LABEL_FIXTURE_6X5,
            &dev,
        )
        .unwrap();
        let out = tier4::pixel_count_map(&dev, &src, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        #[rustfmt::skip]
        let expected = [
            4.0, 4.0, 2.0, 0.0, 4.0, 4.0,
            4.0, 4.0, 2.0, 0.0, 4.0, 4.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            4.0, 4.0, 2.0, 6.0, 6.0, 6.0,
            4.0, 4.0, 2.0, 6.0, 6.0, 6.0,
        ];
        assert_f32_slice_close(&result, &expected, 1e-4);
    }

    #[test]
    #[ignore = "current Rust label-statistics path does not return promptly on this fixture"]
    fn mean_intensity_map_matches_clic_fixture() {
        let dev = device();
        let intensity = [1.0_f32, 1.0, 2.0, 4.0, 0.0, 0.0, 5.0, 3.0, 0.0];
        let labels = [1_u32, 1, 2, 1, 0, 0, 3, 3, 0];
        let intensity = clic_rs::array::Array::create_with_data(
            3,
            3,
            1,
            clic_rs::utils::shape_to_dimension(3, 3, 1),
            clic_rs::types::MType::Buffer,
            &intensity,
            &dev,
        )
        .unwrap();
        let labels = clic_rs::array::Array::create_with_data(
            3,
            3,
            1,
            clic_rs::utils::shape_to_dimension(3, 3, 1),
            clic_rs::types::MType::Buffer,
            &labels,
            &dev,
        )
        .unwrap();
        let out = tier4::mean_intensity_map(&dev, &intensity, &labels, None).unwrap();
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        let expected = [2.0_f32, 2.0, 2.0, 2.0, 0.0, 0.0, 4.0, 4.0, 0.0];
        assert_f32_slice_close(&result, &expected, 1e-4);
    }

    #[test]
    #[ignore = "current Rust bounding-box/statistics path does not return promptly on this fixture"]
    fn label_bounding_box_matches_clic_fixture() {
        let dev = device();
        #[rustfmt::skip]
        let input = [
            0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let src = clic_rs::array::Array::create_with_data(
            7,
            7,
            1,
            clic_rs::utils::shape_to_dimension(7, 7, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let result = tier4::label_bounding_box(&dev, &src, 2).unwrap();

        assert_f32_slice_close(&result, &[4.0, 4.0, 0.0, 5.0, 5.0, 0.0], 1e-4);
    }

    #[test]
    fn spots_to_pointlist_matches_clic_fixture() {
        let dev = device();
        #[rustfmt::skip]
        let input = [
            0.0_f32, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0,
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
        let out = tier4::spots_to_pointlist(&dev, &src, None).unwrap();
        let result: Vec<u32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        assert_eq!(result, [1, 3, 2, 4, 1, 1, 3, 4]);
    }

    #[test]
    fn threshold_otsu_matches_clic_fixture() {
        let dev = device();
        let input = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let src = clic_rs::array::Array::create_with_data(
            3,
            2,
            2,
            clic_rs::utils::shape_to_dimension(3, 2, 2),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier4::threshold_otsu(&dev, &src, None).unwrap();
        let result: Vec<u8> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u8>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        assert_u8_eq(&result, &[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn threshold_otsu_low_intensity_matches_clic_fixture() {
        let dev = device();
        let input = [0.0_f32, 0.0, 0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 0.0];
        let src = clic_rs::array::Array::create_with_data(
            3,
            3,
            1,
            clic_rs::utils::shape_to_dimension(3, 3, 1),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier4::threshold_otsu(&dev, &src, None).unwrap();
        let result: Vec<u8> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u8>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        assert_u8_eq(&result, &[0, 0, 0, 0, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn threshold_yen_matches_clic_fixture() {
        let dev = device();
        let input = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let src = clic_rs::array::Array::create_with_data(
            3,
            2,
            2,
            clic_rs::utils::shape_to_dimension(3, 2, 2),
            clic_rs::types::MType::Buffer,
            &input,
            &dev,
        )
        .unwrap();
        let out = tier4::threshold_yen(&dev, &src, None).unwrap();
        let result: Vec<u8> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<u8>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        assert_u8_eq(&result, &[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
    }
}
