// Performance comparison: fused preproc::letterboxToTensor vs OpenCV GPU preprocessing.
//
// Measures latency of both approaches on the same image and spec, runs inference with
// each to validate numerical equivalence, and prints a comparison table.
//
// Usage: preproc_comparison <model.onnx|engine> <image> [outW=640 outH=640 iters=500 warmup=100]
//
// The example depends on OpenCV (core + imgproc + cudaimgproc modules). It does NOT use
// trtcpp::opencv interop -- it calls raw OpenCV CUDA functions directly for a fair,
// library-independent comparison.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <tensorrt_cpp_api/all.h>
#include <tensorrt_cpp_api/preproc.h>

#include "../common/image_io.h"

using namespace trtcpp;

namespace chrono = std::chrono;

// -----------------------------------------------------------------------
// Timing helper
// -----------------------------------------------------------------------
struct Timing {
    double min = 1e9;
    double max = 0;
    double avg = 0;
    int iters = 0;
};

static Timing timeIt(int iters, const auto &fn) {
    Timing t;
    t.iters = iters;
    double sum = 0;
    for (int i = 0; i < iters; ++i) {
        cudaDeviceSynchronize();
        auto t0 = chrono::steady_clock::now();
        fn();
        cudaDeviceSynchronize();
        auto t1 = chrono::steady_clock::now();
        double ms = chrono::duration<double, std::milli>(t1 - t0).count();
        sum += ms;
        if (ms < t.min) t.min = ms;
        if (ms > t.max) t.max = ms;
    }
    t.avg = sum / iters;
    return t;
}

// -----------------------------------------------------------------------
// OpenCV preprocessing pipeline
// Performs the same sequence as the fused kernel:
//   letterbox-resize -> BGR/RGB swap -> (pixel-mean)*scale -> HWC->NCHW -> float32
// Intermediate buffers are pre-allocated in OpenCVBuf for a fair comparison
// (avoids per-iteration allocation overhead).
// -----------------------------------------------------------------------
struct OpenCVBuf {
    cv::cuda::GpuMat resized;
    cv::cuda::GpuMat padded;
    std::vector<cv::cuda::GpuMat> planes;
};

static Status opencvPreproc(const TensorView &src, TensorView dst, const preproc::PreprocSpec &spec,
                            OpenCVBuf &buf, const Stream &) {
    const int H = static_cast<int>(src.shape()[0]);
    const int W = static_cast<int>(src.shape()[1]);
    const int C = static_cast<int>(src.shape()[2]);
    const int outC = static_cast<int>(dst.shape()[1]);
    const int outH = static_cast<int>(dst.shape()[2]);
    const int outW = static_cast<int>(dst.shape()[3]);

    // Wrap the raw uint8 device buffer as a GpuMat (HWC, continuous).
    cv::cuda::GpuMat gpuSrc(H, W, CV_8UC3, src.data());

    // --- resize (with optional letterbox) ---
    int newW = outW, newH = outH;
    if (spec.keepAspectRatioPad) {
        const float scale = std::min(static_cast<float>(outW) / W, static_cast<float>(outH) / H);
        newW = static_cast<int>(std::lround(W * scale));
        newH = static_cast<int>(std::lround(H * scale));
        newW = std::min(newW, outW);
        newH = std::min(newH, outH);
        cv::cuda::resize(gpuSrc, buf.resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
    } else {
        cv::cuda::resize(gpuSrc, buf.resized, cv::Size(outW, outH), 0, 0, cv::INTER_LINEAR);
    }

    // --- BGR<->RGB swap ---
    if (spec.swapRB && C >= 3) {
        cv::cuda::cvtColor(buf.resized, buf.resized, cv::COLOR_BGR2RGB);
    }

    // --- letterbox padding ---
    if (spec.keepAspectRatioPad && (newW < outW || newH < outH)) {
        buf.padded.create(outH, outW, buf.resized.type());
        buf.padded.setTo(cv::Scalar(spec.padValue, spec.padValue, spec.padValue));
        buf.resized.copyTo(buf.padded(cv::Rect(0, 0, newW, newH)));
    } else {
        buf.padded = buf.resized;
    }

    // --- split, normalize per-channel, layout HWC->NCHW ---
    if (static_cast<int>(buf.planes.size()) < C) {
        buf.planes.resize(C);
    }
    cv::cuda::split(buf.padded, buf.planes);

    auto *dstPtr = static_cast<float *>(dst.data());
    for (int c = 0; c < C && c < outC; ++c) {
        const float m = c < 4 ? spec.mean[static_cast<std::size_t>(c)] : 0.0f;
        const float s = c < 4 ? spec.scale[static_cast<std::size_t>(c)] : 1.0f;
        // out = (pixel - mean) * scale  ->  convertTo: alpha=scale, beta=-mean*scale
        cv::cuda::GpuMat outSlice(outH, outW, CV_32F, dstPtr + static_cast<std::size_t>(c) * outH * outW);
        buf.planes[static_cast<std::size_t>(c)].convertTo(outSlice, CV_32F, s, -m * s);
    }

    return Status{};
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.onnx|engine> <image> [outW=640 outH=640 iters=500 warmup=100]\n",
                     argv[0]);
        std::fprintf(stderr, "\nCompares fused preproc::letterboxToTensor vs OpenCV GPU preprocessing latency.\n");
        return 2;
    }

    const std::string modelPath = argv[1];
    const std::string imagePath = argv[2];
    const int outW = argc > 3 ? std::atoi(argv[3]) : 640;
    const int outH = argc > 4 ? std::atoi(argv[4]) : 640;
    const int iters = argc > 5 ? std::atoi(argv[5]) : 500;
    const int warmup = argc > 6 ? std::atoi(argv[6]) : 100;

    // --- load image (stb, shared baseline) ---
    examples::Image img = examples::decodeImage(imagePath);
    if (img.empty()) {
        std::fprintf(stderr, "error: could not read %s\n", imagePath.c_str());
        return 1;
    }
    std::fprintf(stderr, "image: %d x %d, iters=%d warmup=%d\n", img.width, img.height, iters, warmup);

    // --- build / load engine ---
    BuildOptions bo;
    bo.precision = Precision::kFp16;
    bo.engineCacheDir = "engines";
    auto engine = EngineBuilder{}.buildAndLoad(modelPath, bo);
    if (!engine) {
        std::fprintf(stderr, "engine: %s\n", engine.status().message().c_str());
        return 1;
    }

    auto inName = engine->inputNames().front();
    auto outName = engine->outputNames().front();

    // --- allocate IO ---
    Stream stream;
    auto src = examples::uploadHWC(img, stream).value();
    auto dstFused = Tensor::allocate(DType::kFloat32, Shape{1, 3, outH, outW}, Device::kCuda).value();
    auto dstCv = Tensor::allocate(DType::kFloat32, Shape{1, 3, outH, outW}, Device::kCuda).value();
    auto outFused = Tensor::allocate(engine->tensorDType(outName).value(), engine->tensorShape(outName).value(), Device::kCuda).value();
    auto outCv = Tensor::allocate(engine->tensorDType(outName).value(), engine->tensorShape(outName).value(), Device::kCuda).value();
    stream.synchronize();

    // Build a YOLO-style preprocessing spec (letterbox, /255, no mean shift).
    preproc::PreprocSpec spec;
    spec.keepAspectRatioPad = true;
    spec.scale = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f, 1.f};

    // --- warmup ---
    OpenCVBuf cvBuf;
    for (int i = 0; i < warmup; ++i) {
        preproc::letterboxToTensor(src.view(), dstFused.view(), spec, stream);
        stream.synchronize();
        opencvPreproc(src.view(), dstCv.view(), spec, cvBuf, stream);
        cudaDeviceSynchronize();
    }

    // --- benchmark fused ---
    auto tFused = timeIt(iters, [&] {
        preproc::letterboxToTensor(src.view(), dstFused.view(), spec, stream);
        stream.synchronize();
    });

    // --- benchmark OpenCV ---
    auto tCv = timeIt(iters, [&] {
        opencvPreproc(src.view(), dstCv.view(), spec, cvBuf, stream);
        cudaDeviceSynchronize();
    });

    // --- numerical comparison ---
    auto fusedHost = dstFused.toHost(stream).value();
    auto cvHost = dstCv.toHost(stream).value();
    auto fusedData = fusedHost.as<float>().value();
    auto cvData = cvHost.as<float>().value();

    double mse = 0, maxDiff = 0;
    const auto n = fusedHost.nbytes() / sizeof(float);
    for (size_t i = 0; i < n; ++i) {
        const double d = std::abs(static_cast<double>(fusedData[i]) - static_cast<double>(cvData[i]));
        mse += d * d;
        if (d > maxDiff) maxDiff = d;
    }
    mse /= static_cast<double>(n);

    // --- inference with both to check end-to-end ---
    auto doInfer = [&](Tensor &input, Tensor &output) -> Status {
        return engine->enqueue({{inName, input.view()}}, {{outName, output.view()}}, stream);
    };

    cudaDeviceSynchronize(); // drain any pending work
    auto tInfFused = timeIt(iters, [&] {
        doInfer(dstFused, outFused);
        stream.synchronize();
    });
    auto tInfCv = timeIt(iters, [&] {
        doInfer(dstCv, outCv);
        stream.synchronize();
    });

    // Compare inference outputs
    auto outFusedHost = outFused.toHost(stream).value();
    auto outCvHost = outCv.toHost(stream).value();
    auto outFusedData = outFusedHost.as<float>().value();
    auto outCvData = outCvHost.as<float>().value();

    double outMse = 0, outMaxDiff = 0;
    const auto nOut = outFusedHost.nbytes() / sizeof(float);
    for (size_t i = 0; i < nOut; ++i) {
        const double d = std::abs(static_cast<double>(outFusedData[i]) - static_cast<double>(outCvData[i]));
        outMse += d * d;
        if (d > outMaxDiff) outMaxDiff = d;
    }
    outMse /= static_cast<double>(nOut);

    // --- print results ---
    auto printRow = [](const char *name, const Timing &t) {
        std::printf("%-12s  %8.4f  %8.4f  %8.4f  %8.0f\n", name, t.avg, t.min, t.max, 1000.0 / t.avg);
    };

    std::printf("\n========== Preprocessing Performance ==========\n");
    std::printf("Input: %d x %d  ->  Output: %d x %d\n", img.width, img.height, outW, outH);
    std::printf("Spec: letterbox=%d  swapRB=%d  scale=1/255  iters=%d\n\n",
                spec.keepAspectRatioPad, spec.swapRB, iters);
    std::printf("Method        Avg(ms)   Min(ms)   Max(ms)   Img/s\n");
    std::printf("------        -------   -------   -------   -----\n");
    printRow("Fused", tFused);
    printRow("OpenCV", tCv);
    std::printf("\nOpenCV is %.1f x slower than fused (preproc only)\n", tCv.avg / tFused.avg);
    std::printf("Preproc output  MSE: %.2e  MaxDiff: %.2e\n", mse, maxDiff);

    std::printf("\n========== Inference (%d iters) ==========\n", iters);
    std::printf("Method        Avg(ms)   Min(ms)   Max(ms)   Inf/s\n");
    std::printf("------        -------   -------   -------   -----\n");
    printRow("Fused", tInfFused);
    printRow("OpenCV", tInfCv);
    std::printf("\nInference output  MSE: %.2e  MaxDiff: %.2e\n", outMse, outMaxDiff);

    return 0;
}
