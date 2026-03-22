# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This repository is a **pure Rust rewrite of [CLIc](https://github.com/clEsperanto/CLIc)** — a GPU-accelerated image processing library that is the C++ backend for the clEsperanto ecosystem (pyclesperanto, clesperantoj, Fiji/clij3). The C++ reference implementation lives in `./CLIc/`. No C++ FFI bindings — the goal is idiomatic Rust with OpenCL as the primary GPU backend.

## Building the C++ Reference (for comparison)

```bash
# Configure (macOS)
cmake --preset macos-make-release -B /tmp/clic_build

# Build (only the library, not tests — gtest fails on macOS due to cxxabi.h)
cmake --build /tmp/clic_build --parallel 4

# Output: /tmp/clic_build/clic/libCLIc.dylib
```

## Rust Project Commands

```bash
cargo build                          # build the library
cargo test                           # run unit tests (no GPU required)
cargo test --features gpu-tests      # run integration tests (requires OpenCL device)
cargo test <test_name>               # run a single test by name
cargo clippy -- -D warnings          # lint
```

## Architecture Overview

### C++ Reference Structure (CLIc/)

The C++ library has three layers:

1. **Core abstractions** (`CLIc/clic/src/`): `Array`, `Device`, `Backend`, `execution.cpp`
2. **Tier functions** (`CLIc/clic/src/tier0/` through `tier8/`): 158 .cpp files, ~15k lines
3. **OpenCL kernel strings** (fetched from `clij-opencl-kernels` tag 3.5.3 via CMake FetchContent): embedded as C `const char*` in `#include "cle_*.h"` headers

### Key C++ Patterns to Replicate

Every tier function follows the same pattern (see `CLIc/clic/src/tier1/add_images_weighted.cpp`):
```cpp
tier0::create_like(src0, dst, dType::FLOAT);                   // allocate output
KernelInfo kernel = { "kernel_name", kernel::kernel_source };  // name + .cl string
ParameterList params = { {"src0", src0}, {"dst", dst}, ... };
execute(device, kernel, params, {dst->width(), dst->height(), dst->depth()});
```

The `execute()` function in `CLIc/clic/src/execution.cpp` is the critical piece: it generates `#define` preambles encoding array dimensions (`IMAGE_SIZE_src_WIDTH`), data type macros (`USE_FLOAT`, `CONVERT_dst_PIXEL_TYPE`), and read/write accessor macros (`READ_src_IMAGE`, `WRITE_dst_IMAGE`) before compiling the OpenCL kernel. The full program = preamble + backend preamble + kernel source.

### Rust Module Layout

```
src/
  types.rs          # DType, MType enums (must match C++ exactly for kernel preamble generation)
  utils.rs          # sigma2kernelsize, shape_to_dimension — pure functions
  device.rs         # Device trait + OpenCLDevice (holds opencl3::Context + CommandQueue + ProgramCache)
  backend.rs        # Backend trait + OpenCLBackend
  backend_manager.rs # OnceLock<Mutex<BackendManager>> singleton
  cache.rs          # ProgramCache (LRU, 128 entries) + DiskCache (~/.cache/clesperanto/)
  array.rs          # Array struct — Arc<RwLock<Array>>, GPU memory via GpuMemory enum (Buffer/Image)
  execution.rs      # execute(), execute_separable(), generate_defines() — core kernel dispatch
  tier0.rs          # create_like, create_dst, create_one, create_vector
  tier1/            # Elementary operations; math ops generated via macro_rules!
  tier2/ .. tier8/  # Higher-level operations composing lower tiers
kernels/            # Vendored .cl files from clij-opencl-kernels tag 3.5.3
```

### Key Design Decisions

- **`Arc<RwLock<Array>>`** (aliased as `ArrayPtr`) replaces C++ `shared_ptr<Array>`. RwLock needed because allocation and writes mutate the struct while reads (dimensions, dtype) are concurrent.
- **`dyn Backend`** trait object in `BackendManager` enables runtime OpenCL/CUDA switching without changing call sites.
- **Kernel strings** are embedded at compile time via `include_str!("../kernels/cle_xxx.cl")` — zero runtime overhead, self-contained binary.
- **`generate_defines()`** in `execution.rs` must exactly match the C++ logic in `CLIc/clic/src/execution.cpp` (the `#define` preamble determines how kernels read/write arrays).
- **Math operation boilerplate** (~53% of tier code) is eliminated using `macro_rules!`: `unary_math_op!(absolute, "fabs(x)")` generates the full dispatch function.

### Parameter System

The C++ `std::variant<Array::Pointer, float, int, uint, size_t>` becomes:
```rust
pub enum ParameterValue { Array(ArrayPtr), Float(f32), Int(i32), UInt(u32), SizeT(usize) }
pub type ParameterList = Vec<(&'static str, ParameterValue)>;
```

Use `params![("src", src), ("dst", &dst), ("scalar0", 0.5f32)]` helper macro with `From` impls.

### OpenCL Crate

Use `opencl3` (not `ocl`). Key mappings:
- Device/context/queue: `opencl3::device::Device`, `opencl3::context::Context`, `opencl3::command_queue::CommandQueue`
- Memory: `opencl3::memory::Buffer<u8>` for BUFFER type, `opencl3::memory::Image` for IMAGE type
- Kernel build: `opencl3::program::Program::create_and_build_from_source()`

### Disk Cache

Mirrors C++ `DiskCache`: stores compiled program binaries at `~/.cache/clesperanto/<device_hash>/<source_hash>.bin`. Keyed by SHA-256 of the full program source string. Disabled via `CLESPERANTO_NO_CACHE` env var.
