//! Fixture-style parity tests ported from `CLIc/tests/tier1/*.cpp`.
//!
//! These tests are intentionally data-driven where the C++ tests use fixed
//! inputs and expected values. Randomized C++ cases should be converted to
//! deterministic fixtures before being added here.

use approx::assert_abs_diff_eq;
use clic_rs::{
    array::Array, backend_manager::BackendManager, tier1, ArrayPtr, DType, DeviceArc, MType, Result,
};

const W: usize = 10;
const H: usize = 5;
const D: usize = 3;
const N: usize = W * H * D;

fn device() -> DeviceArc {
    BackendManager::get_instance()
        .get_device("", "all")
        .expect("No OpenCL device found; run these tests with a working OpenCL device")
}

fn assert_all_eq(source: &str, actual: &[f32], expected: f32) {
    assert_eq!(actual.len(), N);
    for (index, value) in actual.iter().enumerate() {
        if (*value - expected).abs() > 1e-6 {
            panic!("{source} mismatch at {index}: got {value}, expected {expected}");
        }
    }
}

type UnaryScalarFn = fn(&DeviceArc, &ArrayPtr, Option<ArrayPtr>, f32) -> Result<ArrayPtr>;
type BinaryImageFn = fn(&DeviceArc, &ArrayPtr, &ArrayPtr, Option<ArrayPtr>) -> Result<ArrayPtr>;

struct UnaryScalarCase {
    source: &'static str,
    function_name: &'static str,
    value: f32,
    scalar: f32,
    expected: f32,
    run: UnaryScalarFn,
}

impl UnaryScalarCase {
    fn assert_matches_clic(&self, dev: &DeviceArc) {
        let input = vec![self.value; N];
        let src = clic_rs::array::Array::create_with_data(
            W,
            H,
            D,
            clic_rs::utils::shape_to_dimension(W, H, D),
            clic_rs::types::MType::Buffer,
            &input,
            dev,
        )
        .unwrap();
        let out = (self.run)(dev, &src, None, self.scalar)
            .unwrap_or_else(|err| panic!("{} failed: {err}", self.function_name));
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        assert_all_eq(self.source, &result, self.expected);
    }
}

struct BinaryImageCase {
    source: &'static str,
    function_name: &'static str,
    left: f32,
    right: f32,
    expected: f32,
    run: BinaryImageFn,
}

impl BinaryImageCase {
    fn assert_matches_clic(&self, dev: &DeviceArc) {
        let left = vec![self.left; N];
        let right = vec![self.right; N];
        let src0 = clic_rs::array::Array::create_with_data(
            W,
            H,
            D,
            clic_rs::utils::shape_to_dimension(W, H, D),
            clic_rs::types::MType::Buffer,
            &left,
            dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            W,
            H,
            D,
            clic_rs::utils::shape_to_dimension(W, H, D),
            clic_rs::types::MType::Buffer,
            &right,
            dev,
        )
        .unwrap();
        let out = (self.run)(dev, &src0, &src1, None)
            .unwrap_or_else(|err| panic!("{} failed: {err}", self.function_name));
        let result: Vec<f32> = {
            let lock = out.lock().unwrap();
            let mut data = vec![<f32>::default(); lock.size()];
            lock.read_to(&mut data).unwrap();
            data
        };

        assert_all_eq(self.source, &result, self.expected);
    }
}

#[test]
fn scalar_arithmetic_matches_clic_tier1_fixtures() {
    let dev = device();
    let cases = [
        UnaryScalarCase {
            source: "CLIc/tests/tier1/test_arithmetic_operations.cpp::add_image_and_scalar",
            function_name: "add_image_and_scalar",
            value: 10.0,
            scalar: 5.0,
            expected: 15.0,
            run: tier1::add_image_and_scalar,
        },
        UnaryScalarCase {
            source: "CLIc/tests/tier1/test_arithmetic_operations.cpp::subtract_image_and_scalar",
            function_name: "subtract_image_from_scalar",
            value: 10.0,
            scalar: 5.0,
            expected: -5.0,
            run: tier1::subtract_image_from_scalar,
        },
        UnaryScalarCase {
            source: "CLIc/tests/tier1/test_subtract_image_and_scalar.cpp::execute",
            function_name: "subtract_scalar_from_image",
            value: 10.0,
            scalar: 5.0,
            expected: 5.0,
            run: tier1::subtract_scalar_from_image,
        },
        UnaryScalarCase {
            source: "CLIc/tests/tier1/test_arithmetic_operations.cpp::multiply_image_and_scalar",
            function_name: "multiply_image_and_scalar",
            value: 10.0,
            scalar: 5.0,
            expected: 50.0,
            run: tier1::multiply_image_and_scalar,
        },
        UnaryScalarCase {
            source: "CLIc/tests/tier1/test_arithmetic_operations.cpp::divide_scalar_by_image",
            function_name: "divide_scalar_by_image",
            value: 2.0,
            scalar: 10.0,
            expected: 5.0,
            run: tier1::divide_scalar_by_image,
        },
    ];

    for case in cases {
        case.assert_matches_clic(&dev);
    }
}

#[test]
fn image_arithmetic_matches_clic_tier1_fixtures() {
    let dev = device();
    let cases = [
        BinaryImageCase {
            source: "CLIc/tests/tier1/test_arithmetic_operations.cpp::multiply_images",
            function_name: "multiply_images",
            left: 25.0,
            right: 75.0,
            expected: 1875.0,
            run: tier1::multiply_images,
        },
        BinaryImageCase {
            source: "CLIc/tests/tier1/test_arithmetic_operations.cpp::divide_images",
            function_name: "divide_images",
            left: 10.0,
            right: 2.0,
            expected: 5.0,
            run: tier1::divide_images,
        },
        BinaryImageCase {
            source: "CLIc/tests/tier1/test_maximum_images.cpp::execute",
            function_name: "maximum_images",
            left: 25.0,
            right: 75.0,
            expected: 75.0,
            run: tier1::maximum_images,
        },
        BinaryImageCase {
            source: "CLIc/tests/tier1/test_minimum_images.cpp::execute",
            function_name: "minimum_images",
            left: 25.0,
            right: 75.0,
            expected: 25.0,
            run: tier1::minimum_images,
        },
    ];

    for case in cases {
        case.assert_matches_clic(&dev);
    }
}

#[test]
fn add_images_weighted_matches_clic_arithmetic_fixture() {
    let dev = device();
    let input0 = vec![5.0; N];
    let input1 = vec![4.0; N];
    let src0 = clic_rs::array::Array::create_with_data(
        W,
        H,
        D,
        clic_rs::utils::shape_to_dimension(W, H, D),
        clic_rs::types::MType::Buffer,
        &input0,
        &dev,
    )
    .unwrap();
    let src1 = clic_rs::array::Array::create_with_data(
        W,
        H,
        D,
        clic_rs::utils::shape_to_dimension(W, H, D),
        clic_rs::types::MType::Buffer,
        &input1,
        &dev,
    )
    .unwrap();

    let out = tier1::add_images_weighted(&dev, &src0, &src1, None, 2.0, 3.0).unwrap();
    let result: Vec<f32> = {
        let lock = out.lock().unwrap();
        let mut data = vec![<f32>::default(); lock.size()];
        lock.read_to(&mut data).unwrap();
        data
    };

    assert_all_eq(
        "CLIc/tests/tier1/test_arithmetic_operations.cpp::add_image_weighted",
        &result,
        22.0,
    );
}

#[test]
fn copy_and_copy_cast_match_clic_copy_fixtures() {
    let dev = device();
    let input = vec![10.0; N];
    let src = clic_rs::array::Array::create_with_data(
        W,
        H,
        D,
        clic_rs::utils::shape_to_dimension(W, H, D),
        clic_rs::types::MType::Buffer,
        &input,
        &dev,
    )
    .unwrap();

    let copied = tier1::copy(&dev, &src, None).unwrap();
    let copied_values: Vec<f32> = {
        let lock = copied.lock().unwrap();
        let mut data = vec![<f32>::default(); lock.size()];
        lock.read_to(&mut data).unwrap();
        data
    };

    assert_all_eq(
        "CLIc/tests/tier1/test_copy.cpp::execute",
        &copied_values,
        10.0,
    );

    let dst = Array::create(W, H, D, 3, DType::Uint32, MType::Buffer, &dev).unwrap();
    tier1::copy(&dev, &src, Some(dst.clone())).unwrap();
    let cast_values: Vec<u32> = {
        let lock = dst.lock().unwrap();
        let mut data = vec![<u32>::default(); lock.size()];
        lock.read_to(&mut data).unwrap();
        data
    };

    assert_eq!(cast_values, vec![10; N]);
}

#[test]
fn binary_not_matches_clic_binary_logic_fixture() {
    let dev = device();
    let input = vec![0_u8; N];
    let src = clic_rs::array::Array::create_with_data(
        W,
        H,
        D,
        clic_rs::utils::shape_to_dimension(W, H, D),
        clic_rs::types::MType::Buffer,
        &input,
        &dev,
    )
    .unwrap();

    let out = tier1::binary_not(&dev, &src, None).unwrap();
    let result: Vec<u8> = {
        let lock = out.lock().unwrap();
        let mut data = vec![<u8>::default(); lock.size()];
        lock.read_to(&mut data).unwrap();
        data
    };

    assert_eq!(result, vec![1; N]);
}

#[test]
fn multiply_image_and_position_matches_clic_arithmetic_fixture() {
    let dev = device();
    let input = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, 1.0, //
        2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    let expected = [
        0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 2.0, 3.0, 4.0, //
        0.0, 2.0, 4.0, 6.0, 8.0,
    ];
    let src = clic_rs::array::Array::create_with_data(
        5,
        3,
        1,
        clic_rs::utils::shape_to_dimension(5, 3, 1),
        clic_rs::types::MType::Buffer,
        &input,
        &dev,
    )
    .unwrap();

    let out = tier1::multiply_image_and_position(&dev, &src, None, 0).unwrap();
    let result: Vec<f32> = {
        let lock = out.lock().unwrap();
        let mut data = vec![<f32>::default(); lock.size()];
        lock.read_to(&mut data).unwrap();
        data
    };

    assert_eq!(result.len(), expected.len());
    for (actual, expected) in result.iter().zip(expected) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1e-6);
    }
}
