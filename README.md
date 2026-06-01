[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<p align="center">
  <a href="https://github.com/cyrusbehr/tensorrt-cpp-api">
    <img width="40%" src="images/logo.png" alt="logo">
  </a>

  <h3 align="center">TensorRT C++ API</h3>

  <p align="center">
    <b>A modern, no-throw C++ library for high-performance GPU inference of CNN models with NVIDIA TensorRT — with optional zero-copy Python bindings.</b>
  </p>
</p>

---

`tensorrt_cpp_api` turns an ONNX model into a cached, optimized TensorRT engine and runs it with
a small, leak-free API: name-keyed tensors at the boundary, caller-owned CUDA streams, explicit
host/device transfers, and a `Status`/`Result<T>` error model — no exceptions, no OpenCV or
TensorRT types in the public headers. It targets **TensorRT ≥ 10** (built to the TensorRT 11
surface), CUDA 12, C++20, Linux and Windows/MSVC.

```cpp
#include <tensorrt_cpp_api/all.h>
using namespace trtcpp;

int main() {
    // Build an FP16 engine from ONNX, or load it from the on-disk cache if one is already current.
    BuildOptions opt;
    opt.precision = Precision::kFp16;
    opt.engineCacheDir = "engines";
    auto engine = EngineBuilder{}.buildAndLoad("model.onnx", opt);
    if (!engine) {
        std::fprintf(stderr, "%s\n", engine.status().message().c_str());
        return 1;
    }

    Stream stream;  // owns a CUDA stream — or Stream::wrap(existingHandle) to use yours
    auto input = Tensor::allocate(DType::kFloat32, Shape{1, 3, 640, 640}, Device::kCuda).value();
    // ... fill `input` (e.g. via the fused preproc kernel) ...

    auto output = engine->inferSingle({{engine->inputNames().front(), input.view()}}, stream);
    if (!output) return 1;

    auto host = output->toHost(stream).value();   // explicit D2H + sync; never implicit
    std::span<const float> scores = host.as<float>().value();
    // ... post-process `scores` ...
}
```

## Features

- **Engine cache that's actually safe.** Build-or-load keyed by ONNX content hash + build options
  + TensorRT version + GPU UUID, with a JSON sidecar and atomic writes. A stale cache (changed
  model, options, driver, or GPU) is detected and rebuilt instead of silently misused.
- **Dynamic shapes, done right.** Per-input min/opt/max optimization profiles; `-1`-aware `Shape`;
  one optimization profile per execution context for concurrent dynamic-shape inference.
- **Concurrency.** `EnginePool` leases execution contexts for multi-stream inference; every call
  runs on a caller-provided `Stream`. The engine is thread-compatible; the pool is thread-safe.
- **No leaky abstractions.** No `nvinfer1`, OpenCV, or spdlog types in any public header (PImpl +
  version-gated, generated `build_config.h`). Consumers need TensorRT at runtime, not compile time.
- **Quantization without surprises.** `Precision::kFp16` / `kInt8Qdq` / `kFp8` …; precision is
  version-aware and never a silent no-op (it errors clearly when a mode isn't achievable).
- **Optional fused preprocessing** (`tensorrt_cpp_api::preproc`): one CUDA kernel does
  letterbox-resize → BGR↔RGB → per-channel normalize → HWC→NCHW → cast, no intermediate buffers.
- **Optional zero-copy Python bindings** (`trtcpp`): feed CuPy / PyTorch / Numba GPU arrays in and
  get them back via `__cuda_array_interface__` / DLPack — no host round-trips, GIL released during
  inference. See [`examples/python`](examples/python).
- **Installable.** `cmake --install` produces a `find_package(tensorrt_cpp_api)`-consumable package.
- **Windows/MSVC support.** Full static and shared (DLL) library builds; MSVC-safe overflow checks
  replace `__builtin_mul_overflow`; `DEBUG_POSTFIX` for side-by-side Debug/Release;
  `TRTCPP_API` dllexport/dllimport annotations; stb-based examples run unmodified.

## Performance

Single-stream inference latency (preallocated, zero-copy `enqueue` loop — `examples/benchmark`):

**RTX 3080 Laptop GPU** (TensorRT 10):

| Model | Precision | Latency | Throughput |
|---|---|---|---|
| YOLOv8n | FP16 | 1.07 ms | 937 inf/s |
| YOLOv8n | FP32 | 2.00 ms | 499 inf/s |
| MobileNetV2 | FP16 | 0.31 ms | 3199 inf/s |

**RTX 4060** (TensorRT 10.16, CUDA 13.2, Windows/MSVC):

| Model | Precision | Config | Latency | Throughput |
|---|---|---|---|---|
| YOLOv8n | FP16 | Release | **1.35 ms** | 743 inf/s |
| YOLOv8n | FP16 | Debug | 1.84 ms | 544 inf/s |

The two tables are **not directly comparable** — they differ in several dimensions:

| Factor | RTX 3080 Laptop | RTX 4060 |
|---|---|---|
| GPU arch | Ampere GA104 | Ada Lovelace AD106 |
| TGP / power | ~130 W (mobile) | ~115 W (desktop) |
| TensorRT | 10 (Linux) | 10.16 (Windows) |
| Model | YOLOv8n | YOLOv8n |
| Compiler backend | disabled | enabled (TensorRT 11 default) |
| Build config | Release (GCC) | Release (MSVC) |

The **end-to-end** breakdown (including decode, preproc, infer, post-process, and output) shows a
wider gap between Debug and Release when CPU-side work is significant:

| Example | Release | Debug | Slowdown |
|---|---|---|---|
| Classification (MobileNetV2) | 0.31 s | 0.59 s | 1.9x |
| Detection (YOLOv8n) | 0.34 s | 0.63 s | 1.8x |
| Segmentation (DeepLabV3) | 0.48 s | 1.35 s | 2.8x |
| Benchmark (YOLOv8n, infer only) | 1.35 ms | 1.84 ms | 1.4x |

The benchmark's GPU-bound inference loop is tight, so the Debug penalty is smaller (1.4x). The
end-to-end examples include CPU-heavy post-processing (softmax, NMS, argmax, image encode),
where the debug CRT and unoptimized MSVC codegen add 1.8–2.8x overhead.

Inference time is TensorRT-bound — it is the `enqueueV3` cost of the engine itself, so the wrapper
adds **no** measurable inference overhead. The library's work is everything around that call:
zero-copy name-keyed IO with no per-call allocations or nested-vector copies, a stream-ordered
allocator, and the no-throw `Status`/`Result` API. The Python bindings run the same path within
~13% of C++ (`examples/python/benchmark_parity.py`).

### Image I/O: stb vs OpenCV

The reference examples use vendored **stb** for image decode (zero external dependencies). The
library's optional **OpenCV interop** (`opencv::upload`, `opencv::copyTo`, opencv_interop.h) is
available when `-DTRT_CPP_API_WITH_OPENCV=ON`. Three paths compared (RTX 4060, Release,
810×1080 JPEG, 20 iters):

| Example | stb | OpenCV imread+upload | GpuMat+copyTo |
|---|---|---|---|
| Classification (MobileNetV2) | 8.49 ms | **6.55 ms** | 7.04 ms |
| Detection (YOLOv8n) | 8.14 ms | **8.05 ms** | 8.30 ms |
| Segmentation (DeepLabV3) | 9.00 ms | **8.76 ms** | 9.20 ms |

**I/O-only breakdown:**

| Path | Time |
|---|---|
| stb decode + upload | 5.93 ms |
| OpenCV imread + upload | **5.86 ms** |
| GpuMat upload + copyTo (`cudaMemcpy2DAsync`) | **0.99 ms** |
| Upload only (cached source, either) | ~0.60 ms |

`cv::imread` decodes JPEG ~4% faster than stb on this system. `opencv::copyTo` uses
`cudaMemcpy2DAsync` under the hood, handling pitched (non-continuous) GpuMat rows without a CPU
round-trip — the fastest path when a GpuMat is already resident on the GPU. When starting from
a file, `imread+upload` is the most efficient decode-to-GPU pipeline.

### Preprocessing: fused kernel vs OpenCV GPU operations

The library's optional fused preprocessor (`preproc::letterboxToTensor`) does letterbox resize,
channel swap, per-channel normalize, and HWC→NCHW layout conversion in a **single CUDA kernel**.
The equivalent pipeline with raw OpenCV GPU calls (`cv::cuda::resize`, `cv::cuda::cvtColor`,
`cv::cuda::split`, `convertTo`, padding, and per-channel memory copies) requires **5–6 kernel
launches** and multiple intermediate device buffers. The `examples/preproc_comparison` program
measures this with configurable resolution, iterations, and preprocessing spec.

Measured on RTX 4060 (TensorRT 11, CUDA 13.2, Windows, 640×640 → 8×8, 200 iters):

| Path | Avg latency | Img/s |
|------|------------|-------|
| Fused kernel (1 launch) | **0.017 ms** | 57 727 |
| OpenCV (resize + cvtColor + pad + split + norm+layout) | 0.075 ms | 13 407 |
| **Ratio** | **4.3× faster** | |

The fused kernel is typically **2–5× faster** for preprocessing alone, with the gap widening at
larger output resolutions where the single fused launch avoids proportionally more overhead.

**Full-pipeline comparison** (`examples/pipeline_comparison`) extends the benchmark across all
three task types and breaks down every stage. Measured on RTX 4060 (TensorRT 11, CUDA 13.2,
Windows, 810×1080 JPEG → network input, 200 iters, 50 warmup):

| Metric | Detection (yolov8n) | Classification (mobilenetv2) | Segmentation (deeplabv3) |
|---|---|---|---|
| **Fused total** | 36.94 ms | 12.77 ms | 37.74 ms |
| **OpenCV total** | 37.71 ms | 13.81 ms | 38.35 ms |
| **Pipeline savings** | **0.77 ms** | **1.04 ms** | **0.61 ms** |
| **Fused preproc** | 0.33 ms | 0.31 ms | 0.19 ms |
| **OpenCV preproc** | 0.66 ms | 0.95 ms | 0.22 ms |
| **Preproc speedup** | **2.0×** | **3.0×** | **1.2×** |
| **Upload saved** | 0.61 ms | 0.66 ms | 0.62 ms |
| **Decode (both)** | ~5.6 ms | ~5.6 ms | ~5.5 ms |
| **Inference (both)** | 2.03 ms | 1.27 ms | 1.88 ms |
| **Postproc (both)** | 28.93 ms | 5.48 ms | 30.12 ms |
| **Num. diff (MSE)** | 5.21e+01 | 1.60e+01 | 1.77e+01 |

Per-stage breakdown for detection (fused vs OpenCV):

| Stage | Fused (ms) | OpenCV (ms) | Ratio |
|---|---|---|---|
| Decode | 5.65 | 5.48 | 1.03× |
| Upload to GPU | 0.00 | 0.61 | 0.00× |
| Preprocess | 0.33 | 0.66 | **0.50×** |
| Inference | 2.03 | 2.03 | 1.00× |
| Postprocess | 28.93 | 28.93 | 1.00× |
| **Total** | **36.94** | **37.71** | **0.98×** |

Key takeaways:
- **Fused preproc is 1.2–3.0× faster** than equivalent OpenCV GPU ops
- **Fused path eliminates the upload step** (0.61–0.66 ms saved) by decoding direct to GPU
- **Inference and postprocess are identical** — the engine doesn't know which path produced the input
- **Overall pipeline favors fused** (0.92–0.98×) — preproc + upload savings outweigh decode differences
- **Numerical differences** (MSE 1.6e+01–5.2e+01) from divergent resize interpolation / color conversion; expected and below model-level sensitivity at typical resolutions

Preprocessing-only comparison (isolated, no decode) on the same system (640×640, 500 iters):

| Method | Avg (ms) | Img/s |
|---|---|---|
| Fused | **0.045** | 22 045 |
| OpenCV | 0.091 | 10 955 |
| **Ratio** | **2.0×** | |

```sh
./build/examples/preproc_comparison models/yolov8n.onnx models/test_img.jpg 640 640 500 100
./build/examples/pipeline_comparison detection yolov8n.onnx image.jpg out.jpg 200 50
./build/examples/pipeline_comparison classification mobilenetv2.onnx image.jpg
./build/examples/pipeline_comparison segmentation deeplabv3.onnx image.jpg
```

## Install

TensorRT and CUDA are system/externally provided. In brief:

```sh
cmake -S . -B build -DTRT_CPP_API_BUILD_PREPROC=ON        # add -DTensorRT_DIR=<root> for a tarball
cmake --build build -j$(nproc)
cmake --install build --prefix /opt/trtcpp
```

On Windows with MSVC:

```sh
cmake -S . -B build -DTensorRT_DIR=C:/TensorRT-10.x.x
cmake --build build --config Release
cmake --install build --prefix C:/trtcpp
```

Build as a shared (DLL) library with `-DBUILD_SHARED_LIBS=ON` (uses `TRTCPP_API` annotations from
[`export.h`](include/tensorrt_cpp_api/export.h) for dllexport/dllimport). MSVC Debug builds append
a `d` suffix to library and example binaries (e.g. `tensorrt_cpp_apid.lib`, `classificationd.exe`).

Then in a downstream project:

```cmake
find_package(tensorrt_cpp_api REQUIRED)
target_link_libraries(myapp PRIVATE tensorrt_cpp_api::tensorrt_cpp_api tensorrt_cpp_api::preproc)
```

Python: `pip install .` (builds the `trtcpp` wheel via scikit-build-core). Full details —
apt vs tarball TensorRT, build options, Python — are in [`docs/install.md`](docs/install.md).

## Examples

[`examples/`](examples) has six runnable reference programs, each consuming the installed package:
**classification** (ImageNet top-5), **detection** (YOLOv8n/YOLO11 + NMS), **segmentation** (DeepLabV3),
a **benchmark** (C++ latency baseline), **preproc_comparison** (fused vs OpenCV preproc latency),
and **pipeline_comparison** (full-pipeline breakdown across all three task types). The last two
require OpenCV (`-DTRT_CPP_API_WITH_OPENCV=ON`). `examples/download_models.sh` fetches the models.

| Example | Model | Data | Release | Debug |
|---|---|---|---|---|
| `classification` | MobileNetV2 (FP16) | 810×1080 RGB | **0.31 s** e2e, class 734 @ 44.5% | 0.59 s e2e |
| `detection` | YOLOv8n (FP16) | 810×1080 RGB | **0.34 s** e2e, 5 detections (bus + 4 persons) | 0.63 s e2e |
| `segmentation` | DeepLabV3-MobileNetV3 (FP16) | 810×1080 RGB | **0.48 s** e2e, 21 classes (56.5% background) | 1.35 s e2e |
| `benchmark` (infer only) | YOLOv8n (FP16) | synthetic (preallocated) | **1.35 ms/infer** (743 inf/s) over 100 iters | 1.84 ms/infer (544 inf/s) |
| `preproc_comparison` | (preproc only) | 810×1080 RGB → 640×640 | fused: **0.08 ms** / OpenCV: 0.28 ms | requires OpenCV |
| `pipeline_comparison` | YOLOv8n / MobileNetV2 / DeepLabV3 | 810×1080 RGB | per-stage breakdown, fused 0.6–0.8× total | requires OpenCV |

Run any example from the project root (the binary path varies by build system):

```sh
# Linux (single-config, e.g. Make/Ninja)
./build/examples/classification model.onnx image.jpg
./build/examples/detection model.onnx image.jpg 0.25 out.jpg
./build/examples/segmentation model.onnx image.jpg out.jpg
./build/examples/benchmark model.onnx 200

# Windows (MSVC multi-config, Debug — note the "d" suffix)
.\build\examples\Debug\classificationd.exe .\models\yolov8n.onnx .\models\test_img.jpg
.\build\examples\Debug\detectiond.exe model.onnx image.jpg 0.25 out.jpg
.\build\examples\Debug\segmentationd.exe model.onnx image.jpg out.jpg
.\build\examples\Debug\benchmarkd.exe .\models\yolov8n.onnx .\models\test_img.jpg 0.25 .\models\out.jpg

# Windows (MSVC multi-config, Release — no suffix)
.\build\examples\Release\classification.exe .\models\yolov8n.onnx .\models\test_img.jpg
.\build\examples\Release\detection.exe model.onnx image.jpg 0.25 out.jpg
.\build\examples\Release\segmentation.exe model.onnx image.jpg out.jpg
.\build\examples\Release\benchmark.exe .\models\yolov8n.onnx .\models\test_img.jpg 0.25 .\models\out.jpg
```

## Documentation

- [Quickstart & core concepts](docs/quickstart.md)
- [Installation](docs/install.md)
- [Upgrading from v6](docs/upgrading_from_v6.md)
- API reference: `doxygen Doxyfile` (HTML in `docs/api/`)

## Sister projects

This library is the inference backend for [YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP)
and [YOLOv9-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv9-TensorRT-CPP) (object detection,
segmentation, pose).

## Scope

Linux (primary), CUDA 12, TensorRT ≥ 10, CNN-style vision models. Windows/MSVC is compile-tested
and runtime-verified (static and shared builds, Debug and Release) but CI only covers Linux.
LLM/transformer-specific features are out of scope.

## Contributing

Issues and PRs welcome. Install the hooks with `pre-commit install` (clang-format + cmake-format);
CI runs the build, the CPU test suite, sanitizers, and a Python wheel build. If this project helps
you, a ⭐ is appreciated — connect on [LinkedIn](https://www.linkedin.com/in/cyrus-behroozi/).

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://ltetrel.github.io/"><img src="https://avatars.githubusercontent.com/u/37963074?v=4?s=100" width="100px;" alt="Loic Tetrel"/><br /><sub><b>Loic Tetrel</b></sub></a><br /><a href="https://github.com/cyrusbehr/tensorrt-cpp-api/commits?author=ltetrel" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/thomaskleiven"><img src="https://avatars.githubusercontent.com/u/17145074?v=4?s=100" width="100px;" alt="thomaskleiven"/><br /><sub><b>thomaskleiven</b></sub></a><br /><a href="https://github.com/cyrusbehr/tensorrt-cpp-api/commits?author=thomaskleiven" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/qq978358810"><img src="https://avatars.githubusercontent.com/u/45676681?v=4?s=100" width="100px;" alt="WiCyn"/><br /><sub><b>WiCyn</b></sub></a><br /><a href="https://github.com/cyrusbehr/tensorrt-cpp-api/commits?author=qq978358810" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.

## License

See [LICENSE](LICENSE). Version history is in [CHANGELOG.md](CHANGELOG.md).

<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/cyrusbehr/tensorrt-cpp-api.svg?style=flat-square
[stars-url]: https://github.com/cyrusbehr/tensorrt-cpp-api/stargazers
[issues-shield]: https://img.shields.io/github/issues/cyrusbehr/tensorrt-cpp-api.svg?style=flat-square
[issues-url]: https://github.com/cyrusbehr/tensorrt-cpp-api/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/cyrus-behroozi/
