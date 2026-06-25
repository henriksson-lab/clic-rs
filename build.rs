use std::env;
use std::path::PathBuf;
use std::process::Command;

fn run(mut command: Command) {
    let status = command.status().expect("failed to execute build command");
    if !status.success() {
        panic!("build command failed with status {status}");
    }
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=src/fft_vkfft.c");
    println!("cargo:rerun-if-changed=third_party/VkFFT/vkFFT/vkFFT.h");
    println!("cargo:rerun-if-changed=third_party/VkFFT/vkFFT/vkFFT/");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let object = out_dir.join("fft_vkfft.o");
    let archive = out_dir.join("libclic_vkfft.a");
    let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());

    let mut compile = Command::new(cc);
    compile
        .arg("-std=c99")
        .arg("-O2")
        .arg("-c")
        .arg("src/fft_vkfft.c")
        .arg("-Ithird_party/VkFFT/vkFFT")
        .arg("-DVKFFT_BACKEND=3")
        .arg("-DVKFFT_MAX_FFT_DIMENSIONS=4")
        .arg("-DCL_TARGET_OPENCL_VERSION=120")
        .arg("-o")
        .arg(&object);
    run(compile);

    let mut archive_cmd = Command::new(ar);
    archive_cmd.arg("crs").arg(&archive).arg(&object);
    run(archive_cmd);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=clic_vkfft");
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=OpenCL");
    } else {
        for path in [
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda-12/targets/x86_64-linux/lib",
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
        ] {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={path}");
            }
        }
        println!("cargo:rustc-link-lib=OpenCL");
    }
}
