# clic-rs

A pure Rust rewrite of [CLIc](https://github.com/clEsperanto/CLIc) — the GPU-accelerated image processing backend of the [clEsperanto](https://github.com/clEsperanto) ecosystem (pyclesperanto, clesperantoj, Fiji/clij3).

No C++ FFI. OpenCL via the [`opencl3`](https://crates.io/crates/opencl3) crate.

**more testing pending**

## This is an LLM-mediated faithful (hopefully) translation, not the original code!

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this page](https://rewrites.bio/)
but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back to the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information


## Status

| Tier | Functions | Tests |
|------|-----------|-------|
| tier1 | copy, gaussian_blur, add/subtract/multiply/divide scalars & images, absolute, power, projections, filters | 11 GPU |
| tier2 | absolute_difference, add/subtract images, clip, difference_of_gaussian, morphological ops (opening, closing, top-hat, bottom-hat, std-dev), global reductions | 15 GPU |
| tier3 | mean_of_all_pixels, gamma_correction | 4 GPU |
| tier4 | mean_squared_error | (included above) |
| tier5 | array_equal | 3 GPU |
| tier7 | translate, scale (affine transforms via `affine_transform.cl`) | 5 GPU |

## Usage

```rust
use clic_rs::{BackendManager, array::{push, pull}, tier1};

let device = BackendManager::get().get_device("", "gpu").unwrap();
let src = push(&vec![1.0f32; 100], 10, 10, 1, &device).unwrap();
let dst = tier1::gaussian_blur(&device, &src, None, 1.0, 1.0, 0.0).unwrap();
let result: Vec<f32> = pull(&dst).unwrap();
```

## Benchmarks

Measured on Apple Silicon (Intel GPU, macOS 15.7, OpenCL). Both implementations synchronize the GPU after each operation (`clFinish`). clic-rs benefits from an in-memory LRU kernel program cache, avoiding recompilation on repeated calls.

| Operation | Image size | CLIc (C++) | clic-rs (Rust) |
|-----------|------------|------------|----------------|
| `gaussian_blur` | 64×64 | 3768 µs | 1293 µs |
| `gaussian_blur` | 256×256 | 3321 µs | 1579 µs |
| `gaussian_blur` | 512×512 | 5272 µs | 2833 µs |
| `add_images_weighted` | 64×64 | 1232 µs | 783 µs |
| `add_images_weighted` | 256×256 | 1200 µs | 952 µs |
| `add_images_weighted` | 512×512 | 1456 µs | 1386 µs |
| `mean_of_all_pixels` | 64×64 | 3939 µs | 1116 µs |
| `mean_of_all_pixels` | 256×256 | 2561 µs | 1272 µs |
| `mean_of_all_pixels` | 512×512 | 3505 µs | 1544 µs |

Run benchmarks:

```bash
bash benchmark/run.sh          # compare C++ CLIc vs clic-rs side by side
cargo bench --bench gpu        # Rust only (Criterion HTML report)
```

## Building

Requires an OpenCL runtime (e.g. from your GPU driver or [PoCL](https://portablecl.org) for CPU fallback).

```bash
cargo build
cargo test                        # unit tests (no GPU required)
cargo test --features gpu-tests   # integration tests (requires OpenCL device)
```

## Architecture

```
clic-rs/src/
  execution.rs        # generate_defines() + execute() — core kernel dispatch
  array.rs            # Arc<Mutex<Array>> (ArrayPtr) — GPU memory lifecycle
  backend.rs          # OpenCL backend: allocate, read, write, execute kernels
  device.rs           # OpenCLDevice — context, queue, program cache
  cache.rs            # LRU program cache + SHA-256 disk cache (~/.cache/clesperanto/)
  tier0.rs            # Array creation helpers (create_like, create_one, …)
  tier1/              # Elementary ops: math, filters, projections, blur
  tier2/              # Compositions: morphology, reductions, clipping
  tier3/ … tier7/     # Higher-level compositions
clic-rs/kernels/      # Vendored .cl files from clij-opencl-kernels 3.5.3
CLIc/                 # C++ reference implementation (read-only, for comparison)
```

The execution model mirrors CLIc: `generate_defines()` builds a `#define` preamble encoding array dimensions and data-type macros, which is prepended to the kernel source before compilation. Compiled programs are cached in memory (LRU, 128 entries) and on disk (SHA-256 keyed).

## License

Same as CLIc — see [CLIc/LICENSE](CLIc/LICENSE).
