use std::env;
use std::process;
use std::time::Instant;

use clic_rs::{backend_manager::BackendManager, tier1, tier3, DeviceArc};

fn device() -> DeviceArc {
    BackendManager::get_instance()
        .get_device("", "all")
        .expect("No OpenCL device found")
}

fn make_data(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

fn measure_us<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }

    let samples = 16;
    let mut total = 0.0;
    for _ in 0..samples {
        let start = Instant::now();
        f();
        total += start.elapsed().as_secs_f64() * 1_000_000.0;
    }
    total / samples as f64
}

fn print_result(function: &str, size: &str, mean_us: f64) {
    println!("  {size:<40} {mean_us:9.1} us");
    println!("RESULT {function} {size} {mean_us:.3}");
}

fn bench_gaussian_blur(dev: &DeviceArc) {
    println!("gaussian_blur:");
    for side in [64usize, 256, 512] {
        let n = side * side;
        let data = make_data(n);
        let src = clic_rs::array::Array::create_with_data(
            side,
            side,
            1,
            clic_rs::utils::shape_to_dimension(side, side, 1),
            clic_rs::types::MType::Buffer,
            &data,
            dev,
        )
        .unwrap();
        let size = format!("{side}x{side}");

        let us = measure_us(|| {
            let _out = tier1::gaussian_blur(dev, &src, None, 2.0, 2.0, 0.0).unwrap();
            dev.finish();
        });
        print_result("gaussian_blur", &size, us);
    }
}

fn bench_add_images_weighted(dev: &DeviceArc) {
    println!("add_images_weighted:");
    for side in [64usize, 256, 512] {
        let n = side * side;
        let data = make_data(n);
        let src0 = clic_rs::array::Array::create_with_data(
            side,
            side,
            1,
            clic_rs::utils::shape_to_dimension(side, side, 1),
            clic_rs::types::MType::Buffer,
            &data,
            dev,
        )
        .unwrap();
        let src1 = clic_rs::array::Array::create_with_data(
            side,
            side,
            1,
            clic_rs::utils::shape_to_dimension(side, side, 1),
            clic_rs::types::MType::Buffer,
            &data,
            dev,
        )
        .unwrap();
        let size = format!("{side}x{side}");

        let us = measure_us(|| {
            let _out = tier1::add_images_weighted(dev, &src0, &src1, None, 0.5, 0.5).unwrap();
            dev.finish();
        });
        print_result("add_images_weighted", &size, us);
    }
}

fn bench_mean_of_all_pixels(dev: &DeviceArc) {
    println!("mean_of_all_pixels:");
    for side in [64usize, 256, 512] {
        let n = side * side;
        let data = make_data(n);
        let src = clic_rs::array::Array::create_with_data(
            side,
            side,
            1,
            clic_rs::utils::shape_to_dimension(side, side, 1),
            clic_rs::types::MType::Buffer,
            &data,
            dev,
        )
        .unwrap();
        let size = format!("{side}x{side}");

        let us = measure_us(|| {
            let _mean = tier3::mean_of_all_pixels(dev, &src).unwrap();
        });
        print_result("mean_of_all_pixels", &size, us);
    }
}

fn bench_push_pull(dev: &DeviceArc) {
    println!("push_pull:");
    for side in [64usize, 256, 512] {
        let n = side * side;
        let data = make_data(n);
        let size = format!("{side}x{side}");

        let us = measure_us(|| {
            let arr = clic_rs::array::Array::create_with_data(
                side,
                side,
                1,
                clic_rs::utils::shape_to_dimension(side, side, 1),
                clic_rs::types::MType::Buffer,
                &data,
                dev,
            )
            .unwrap();
            let _out: Vec<f32> = {
                let lock = arr.lock().unwrap();
                let mut data = vec![<f32>::default(); lock.size()];
                lock.read_to(&mut data).unwrap();
                data
            };
        });
        print_result("push_pull", &size, us);
    }
}

fn list_cases() {
    println!("gaussian_blur");
    println!("add_images_weighted");
    println!("mean_of_all_pixels");
    println!("push_pull");
}

fn main() {
    let Some(name) = env::args().nth(1) else {
        eprintln!("usage: parity_bench <benchmark_name|--list>");
        process::exit(1);
    };
    if name == "--list" {
        list_cases();
        return;
    }

    let dev = device();
    match name.as_str() {
        "gaussian_blur" => bench_gaussian_blur(&dev),
        "add_images_weighted" => bench_add_images_weighted(&dev),
        "mean_of_all_pixels" => bench_mean_of_all_pixels(&dev),
        "push_pull" => bench_push_pull(&dev),
        _ => {
            eprintln!("unknown benchmark: {name}");
            process::exit(1);
        }
    }
}
