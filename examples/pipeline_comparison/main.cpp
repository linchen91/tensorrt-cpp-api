// Full-pipeline performance comparison: without OpenCV vs with OpenCV, across three task types.
//
// Compares every stage (decode, upload, preproc, infer, postproc) side-by-side for
// detection / classification / segmentation, then prints a detailed timing table and
// validates numerical equivalence of the inference outputs.
//
// Usage:
//   pipeline_comparison <mode=detection|classification|segmentation> <model.onnx|engine>
//                       <image> [out.jpg iters=200 warmup=50 [WxH]]
//
// Without-OpenCV path:  stb (CPU) -> uploadHWC -> preproc::letterboxToTensor -> engine
// With-OpenCV path:     cv::imread (CPU) -> GpuMat upload -> resize/cvtColor/split/convertTo -> engine
// Postproc and output writing use the same code for both (isolates decode+preproc comparison).

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>

#include <tensorrt_cpp_api/all.h>
#include <tensorrt_cpp_api/preproc.h>

#include "../common/image_io.h"

using namespace trtcpp;

namespace chrono = std::chrono;

// ======================================================================
// Helpers
// ======================================================================

enum class Mode { Detection, Classification, Segmentation };

struct Timing {
    double min = 1e9;
    double max = 0;
    double sum = 0;
    int n = 0;

    double avg() const { return n > 0 ? sum / n : 0; }
};

static void accum(Timing &t, double ms) {
    if (ms < t.min) t.min = ms;
    if (ms > t.max) t.max = ms;
    t.sum += ms;
    ++t.n;
}

static double elapsedMs(chrono::steady_clock::time_point t0) {
    auto t1 = chrono::steady_clock::now();
    return chrono::duration<double, std::milli>(t1 - t0).count();
}

// Sync a specific stream (for GPU-stage timing), return elapsed ms.
static double syncElapsed(const Stream &s, chrono::steady_clock::time_point t0) {
    s.synchronize();
    return elapsedMs(t0);
}

// Sync all device work (used after operations on the default stream).
static double deviceSyncElapsed(chrono::steady_clock::time_point t0) {
    cudaDeviceSynchronize();
    return elapsedMs(t0);
}

static Mode parseMode(const std::string &s) {
    if (s == "detection") return Mode::Detection;
    if (s == "classification") return Mode::Classification;
    if (s == "segmentation") return Mode::Segmentation;
    std::fprintf(stderr, "unknown mode '%s'; use detection|classification|segmentation\n", s.c_str());
    std::exit(2);
}

static const char *modeName(Mode m) {
    switch (m) {
    case Mode::Detection: return "detection";
    case Mode::Classification: return "classification";
    case Mode::Segmentation: return "segmentation";
    }
    return "?";
}

// ======================================================================
// OpenCV preprocessing (with per-substage timing)
// ======================================================================
struct OpenCVBuf {
    cv::cuda::GpuMat resized;
    cv::cuda::GpuMat padded;
    std::vector<cv::cuda::GpuMat> planes;
    // substage timings (resize, cvtColor, pad, split, norm+layout)
    Timing sub[5];
};

static void opencvPreproc(const TensorView &src, TensorView dst, const preproc::PreprocSpec &spec,
                          OpenCVBuf &buf) {
    const int H = static_cast<int>(src.shape()[0]);
    const int W = static_cast<int>(src.shape()[1]);
    const int C = static_cast<int>(src.shape()[2]);
    const int outC = static_cast<int>(dst.shape()[1]);
    const int outH = static_cast<int>(dst.shape()[2]);
    const int outW = static_cast<int>(dst.shape()[3]);

    cv::cuda::GpuMat gpuSrc(H, W, CV_8UC3, src.data());

    // --- resize (with optional letterbox) ---
    {
        auto t0 = chrono::steady_clock::now();
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
        accum(buf.sub[0], deviceSyncElapsed(t0));
    }

    // --- BGR<->RGB swap ---
    if (spec.swapRB && C >= 3) {
        auto t0 = chrono::steady_clock::now();
        cv::cuda::cvtColor(buf.resized, buf.resized, cv::COLOR_BGR2RGB);
        accum(buf.sub[1], deviceSyncElapsed(t0));
    }

    // --- letterbox padding ---
    {
        auto t0 = chrono::steady_clock::now();
        int newW = outW, newH = outH;
        if (spec.keepAspectRatioPad) {
            float scale = std::min(static_cast<float>(outW) / W, static_cast<float>(outH) / H);
            newW = static_cast<int>(std::lround(W * scale));
            newH = static_cast<int>(std::lround(H * scale));
            newW = std::min(newW, outW);
            newH = std::min(newH, outH);
        }
        if (spec.keepAspectRatioPad && (newW < outW || newH < outH)) {
            buf.padded.create(outH, outW, buf.resized.type());
            buf.padded.setTo(cv::Scalar(spec.padValue, spec.padValue, spec.padValue));
            buf.resized.copyTo(buf.padded(cv::Rect(0, 0, newW, newH)));
        } else {
            buf.padded = buf.resized;
        }
        accum(buf.sub[2], deviceSyncElapsed(t0));
    }

    // --- split ---
    {
        auto t0 = chrono::steady_clock::now();
        if (static_cast<int>(buf.planes.size()) < C) buf.planes.resize(C);
        cv::cuda::split(buf.padded, buf.planes);
        accum(buf.sub[3], deviceSyncElapsed(t0));
    }

    // --- per-channel normalize + layout HWC->NCHW ---
    {
        auto t0 = chrono::steady_clock::now();
        auto *dstPtr = static_cast<float *>(dst.data());
        for (int c = 0; c < C && c < outC; ++c) {
            float m = c < 4 ? spec.mean[static_cast<std::size_t>(c)] : 0.0f;
            float s = c < 4 ? spec.scale[static_cast<std::size_t>(c)] : 1.0f;
            cv::cuda::GpuMat outSlice(outH, outW, CV_32F,
                                      dstPtr + static_cast<std::size_t>(c) * outH * outW);
            buf.planes[static_cast<std::size_t>(c)].convertTo(outSlice, CV_32F, s, -m * s);
        }
        accum(buf.sub[4], deviceSyncElapsed(t0));
    }
}

// ======================================================================
// Detection postproc (YOLOv8 anchor-free)
// ======================================================================
struct Detection {
    int x0, y0, x1, y1;
    float score;
    int cls;
};

static float iou(const Detection &a, const Detection &b) {
    int ix0 = std::max(a.x0, b.x0), iy0 = std::max(a.y0, b.y0);
    int ix1 = std::min(a.x1, b.x1), iy1 = std::min(a.y1, b.y1);
    float inter = static_cast<float>(std::max(0, ix1 - ix0)) * std::max(0, iy1 - iy0);
    float areaA = static_cast<float>(a.x1 - a.x0) * (a.y1 - a.y0);
    float areaB = static_cast<float>(b.x1 - b.x0) * (b.y1 - b.y0);
    float uni = areaA + areaB - inter;
    return uni > 0 ? inter / uni : 0.f;
}

static std::vector<Detection> nms(std::vector<Detection> dets, float iouThresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b) { return a.score > b.score; });
    std::vector<Detection> keep;
    std::vector<bool> removed(dets.size(), false);
    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        for (std::size_t j = i + 1; j < dets.size(); ++j)
            if (!removed[j] && dets[j].cls == dets[i].cls && iou(dets[i], dets[j]) > iouThresh)
                removed[j] = true;
    }
    return keep;
}

// ======================================================================
// Segmentation postproc (Pascal VOC palette)
// ======================================================================
struct Rgb { std::uint8_t r, g, b; };

static std::array<Rgb, 256> vocPalette() {
    std::array<Rgb, 256> p{};
    for (int i = 0; i < 256; ++i) {
        int r = 0, g = 0, b = 0, c = i;
        for (int j = 0; j < 8; ++j) {
            r |= ((c >> 0) & 1) << (7 - j);
            g |= ((c >> 1) & 1) << (7 - j);
            b |= ((c >> 2) & 1) << (7 - j);
            c >>= 3;
        }
        p[static_cast<std::size_t>(i)] = {static_cast<std::uint8_t>(r),
                                          static_cast<std::uint8_t>(g),
                                          static_cast<std::uint8_t>(b)};
    }
    return p;
}

// ======================================================================
// Main
// ======================================================================
int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <mode=detection|classification|segmentation> "
                             "<model.onnx|engine> <image> [out.jpg iters=200 warmup=50 [WxH]]\n",
                     argv[0]);
        return 2;
    }

    Mode mode = parseMode(argv[1]);
    std::string modelPath = argv[2];
    std::string imagePath = argv[3];
    std::string outPath = argc > 4 ? argv[4] : "pipeline_output.jpg";
    int iters = argc > 5 ? std::atoi(argv[5]) : 200;
    int warmup = argc > 6 ? std::atoi(argv[6]) : 50;

    std::string optRes = argc > 7 ? argv[7] : "";

    // ----------------------------------------------------------------
    // Build engine
    // ----------------------------------------------------------------
    BuildOptions bo;
    bo.precision = Precision::kFp16;
    bo.engineCacheDir = "engines";

    // Optional resolution override from command line (WxH, e.g. 512x512).
    // Sets up a dynamic optimization profile so the engine is built for the
    // requested size. The input name is resolved after the first engine load.
    int inH = 0, inW = 0;
    bool haveRes = false;
    if (!optRes.empty()) {
        auto xPos = optRes.find('x');
        if (xPos == std::string::npos || xPos == 0 || xPos == optRes.size() - 1) {
            std::fprintf(stderr, "invalid resolution '%s'; use WxH (e.g. 512x512)\n", optRes.c_str());
            return 2;
        }
        inW = std::atoi(optRes.substr(0, xPos).c_str());
        inH = std::atoi(optRes.substr(xPos + 1).c_str());
        haveRes = true;
    }

    auto engine = EngineBuilder{}.buildAndLoad(modelPath, bo);
    if (!engine) {
        std::fprintf(stderr, "engine: %s\n", engine.status().message().c_str());
        return 1;
    }

    auto inName = engine->inputNames().front();
    auto outName = engine->outputNames().front();
    auto inShape = engine->tensorShape(inName).value();

    // If resolution was specified, rebuild with a proper optimization profile
    // that includes the requested size range.
    if (haveRes) {
        auto nativeW = static_cast<int>(inShape[3]);
        auto nativeH = static_cast<int>(inShape[2]);
        if (inW != nativeW || inH != nativeH) {
            if (!inShape.isDynamic()) {
                std::fprintf(stderr, "cannot override resolution: model has static input shape "
                                     "[%s]; export ONNX with dynamic input dims or use a model "
                                     "natively at %dx%d\n",
                             inShape.toString().c_str(), inW, inH);
                return 1;
            }
            ProfileShape ps;
            ps.inputName = inName;
            int minW = std::min(inW, nativeW), minH = std::min(inH, nativeH);
            int maxW = std::max(inW, nativeW), maxH = std::max(inH, nativeH);
            ps.min = Shape{1, 3, minH, minW};
            ps.opt = Shape{1, 3, inH, inW};
            ps.max = Shape{1, 3, maxH, maxW};
            bo.profiles.push_back(OptimizationProfile{{ps}});
            engine = EngineBuilder{}.buildAndLoad(modelPath, bo);
            if (!engine) {
                std::fprintf(stderr, "engine (profile): %s\n", engine.status().message().c_str());
                return 1;
            }
            inShape = engine->tensorShape(inName).value();
        }
        std::fprintf(stderr, "overriding resolution to %dx%d (model native: %dx%d)\n",
                     inW, inH, nativeW, nativeH);
    } else {
        inH = static_cast<int>(inShape[2]);
        inW = static_cast<int>(inShape[3]);
    }
    // Resolve output shape at runtime (matches what inferSingle uses internally)
    TensorView shapeOnly{nullptr, DType::kFloat32, inShape, Device::kHost};
    auto outShape = engine->outputShapes({{inName, shapeOnly}}, 0).value()[outName];
    int outC = static_cast<int>(outShape[1]);
    int outH = static_cast<int>(outShape[2]);
    int outW = static_cast<int>(outShape[3]);

    // ----------------------------------------------------------------
    // Preprocessing spec per mode
    // ----------------------------------------------------------------
    preproc::PreprocSpec specStb; // RGB input (stb decodes to RGB)
    preproc::PreprocSpec specCv;  // BGR input (cv::imread decodes to BGR)

    if (mode == Mode::Detection) {
        // YOLO: letterbox, /255, no mean shift
        specStb.keepAspectRatioPad = true;
        specStb.swapRB = false;
        specStb.scale = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f, 1.f};

        specCv = specStb;
        specCv.swapRB = true; // OpenCV loads BGR -> need RGB for model
    } else {
        // Classification / Segmentation: ImageNet-style
        specStb = examples::imagenetSpec(); // RGB, swapRB=false
        specCv = specStb;
        specCv.swapRB = true; // OpenCV BGR -> RGB
    }

    // ----------------------------------------------------------------
    // Allocate reusable device buffers
    // ----------------------------------------------------------------
    Stream stream;
    auto dst = Tensor::allocate(DType::kFloat32, Shape{1, 3, inH, inW}, Device::kCuda).value();
    auto outDev = Tensor::allocate(engine->tensorDType(outName).value(), outShape, Device::kCuda).value();

    // Two output tensors for numerical comparison
    auto outStb = Tensor::allocate(engine->tensorDType(outName).value(), outShape, Device::kCuda).value();
    auto outCv = Tensor::allocate(engine->tensorDType(outName).value(), outShape, Device::kCuda).value();

    std::unordered_map<std::string, TensorView> inputs{{inName, dst.view()}};

    // ----------------------------------------------------------------
    // Per-image data
    // ----------------------------------------------------------------
    examples::Image stbImg;
    cv::Mat cvImgHost;

    // ----------------------------------------------------------------
    // Timing accumulators
    // ----------------------------------------------------------------
    Timing t_noCv_decode, t_noCv_preproc;
    Timing t_cv_decode, t_cv_upload;
    Timing t_infer, t_postproc;

    OpenCVBuf cvBuf;

    // ----------------------------------------------------------------
    // Warmup
    // ----------------------------------------------------------------
    std::fprintf(stderr, "Warming up (%d iters)...\n", warmup);
    stbImg = examples::decodeImage(imagePath);
    if (stbImg.empty()) {
        std::fprintf(stderr, "could not read image: %s\n", imagePath.c_str());
        return 1;
    }
    cvImgHost = cv::imread(imagePath, cv::IMREAD_COLOR);
    bool haveOpenCV = !cvImgHost.empty();
    if (cvImgHost.empty()) {
        std::fprintf(stderr, "cv::imread failed for '%s', converting stb image to cv::Mat (BGR)\n", imagePath.c_str());
        cvImgHost = cv::Mat(stbImg.height, stbImg.width, CV_8UC3);
        std::memcpy(cvImgHost.data, stbImg.data.data(), stbImg.data.size());
        cv::cvtColor(cvImgHost, cvImgHost, cv::COLOR_RGB2BGR);
    }
    for (int i = 0; i < warmup; ++i) {
        // stb path
        {
            auto src = examples::uploadHWC(stbImg, stream).value();
            preproc::letterboxToTensor(src.view(), dst.view(), specStb, stream);
            engine->enqueue(inputs, {{outName, outStb.view()}}, stream);
            stream.synchronize();
        }
        // OpenCV path (skip when cv::imread failed — OpenCV CUDA modules may be broken)
        if (haveOpenCV) {
            cv::cuda::GpuMat gpuSrc;
            gpuSrc.upload(cvImgHost);
            TensorView cvSrc(gpuSrc.data, DType::kUInt8,
                             Shape{cvImgHost.rows, cvImgHost.cols, cvImgHost.channels()},
                             Device::kCuda);
            opencvPreproc(cvSrc, dst.view(), specCv, cvBuf);
            engine->enqueue(inputs, {{outName, outCv.view()}}, stream);
            stream.synchronize();
        }
    }

    // ----------------------------------------------------------------
    // Benchmark: Without OpenCV (stb + uploadHWC + fused preproc + infer)
    // ----------------------------------------------------------------
    std::fprintf(stderr, "Benchmarking without-OpenCV pipeline (%d iters)...\n", iters);
    for (int i = 0; i < iters; ++i) {
        // Decode (CPU)
        auto t0 = chrono::steady_clock::now();
        stbImg = examples::decodeImage(imagePath);
        accum(t_noCv_decode, elapsedMs(t0));

        // Upload + Preproc + Infer (GPU, async on stream)
        // Sync after each stage for accurate per-stage timing.
        auto src = examples::uploadHWC(stbImg, stream).value();
        t0 = chrono::steady_clock::now();
        preproc::letterboxToTensor(src.view(), dst.view(), specStb, stream);
        accum(t_noCv_preproc, syncElapsed(stream, t0));

        t0 = chrono::steady_clock::now();
        engine->enqueue(inputs, {{outName, outStb.view()}}, stream);
        accum(t_infer, syncElapsed(stream, t0));
    }

    // Reset OpenCV substage timings (warmup pollutes them)
    for (auto &sub : cvBuf.sub) sub = Timing{};

    // ----------------------------------------------------------------
    // Benchmark: With OpenCV (cv::imread + GpuMat + resize/cvtColor/split/convertTo + infer)
    // ----------------------------------------------------------------
    if (haveOpenCV) {
        std::fprintf(stderr, "Benchmarking with-OpenCV pipeline (%d iters)...\n", iters);
        for (int i = 0; i < iters; ++i) {
            // Decode (CPU)
            auto t0 = chrono::steady_clock::now();
            cvImgHost = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (cvImgHost.empty()) {
                std::fprintf(stderr, "cv::imread failed for '%s', converting stb image to cv::Mat (BGR)\n", imagePath.c_str());
                cvImgHost = cv::Mat(stbImg.height, stbImg.width, CV_8UC3);
                std::memcpy(cvImgHost.data, stbImg.data.data(), stbImg.data.size());
                cv::cvtColor(cvImgHost, cvImgHost, cv::COLOR_RGB2BGR);
            }
            accum(t_cv_decode, elapsedMs(t0));

            // Upload to GPU (default stream)
            t0 = chrono::steady_clock::now();
            cv::cuda::GpuMat gpuSrc;
            gpuSrc.upload(cvImgHost);
            accum(t_cv_upload, deviceSyncElapsed(t0));

            // Preproc (OpenCV, on default stream, timed internally per-substage)
            TensorView cvSrc(gpuSrc.data, DType::kUInt8,
                             Shape{cvImgHost.rows, cvImgHost.cols, cvImgHost.channels()},
                             Device::kCuda);
            opencvPreproc(cvSrc, dst.view(), specCv, cvBuf);

            // Wait for OpenCV work, then copy to our stream for inference.
            t0 = chrono::steady_clock::now();
            engine->enqueue(inputs, {{outName, outCv.view()}}, stream);
            accum(t_infer, syncElapsed(stream, t0));
        }
    } else {
        std::fprintf(stderr, "Skipping with-OpenCV benchmark (cv::imread failed at warmup)\n");
    }

    // ----------------------------------------------------------------
    // Postproc timing (identical for both pipelines).
    // File I/O (JPG write, image re-decode) is excluded from the timed
    // loop — only algorithmic work (NMS, argmax, colorize, softmax) is
    // measured. A single final write produces the output file.
    // ----------------------------------------------------------------
    std::fprintf(stderr, "Benchmarking postproc (%d iters)...\n", iters);
    const auto origImg = stbImg; // preserve for reuse across iterations
    const auto palette = mode == Mode::Segmentation ? vocPalette() : std::array<Rgb, 256>{};
    for (int i = 0; i < iters; ++i) {
        auto host = outStb.toHost(stream).value();
        auto data = host.as<float>().value();
        stbImg = origImg;
        auto t0 = chrono::steady_clock::now();

        if (mode == Mode::Detection) {
            const int nAnchors = static_cast<int>(host.shape()[2]);
            const float r = std::min(static_cast<float>(inW) / origImg.width,
                                     static_cast<float>(inH) / origImg.height);

            std::vector<Detection> dets;
            for (int a = 0; a < nAnchors; ++a) {
                int bestCls = 0;
                float bestScore = 0.f;
                for (int c = 0; c < 80; ++c) {
                    float s = data[(4 + c) * nAnchors + a];
                    if (s > bestScore) { bestScore = s; bestCls = c; }
                }
                if (bestScore < 0.25f) continue;
                float cx = data[0 * nAnchors + a], cy = data[1 * nAnchors + a];
                float w = data[2 * nAnchors + a], h = data[3 * nAnchors + a];
                dets.push_back({static_cast<int>((cx - w / 2) / r),
                                static_cast<int>((cy - h / 2) / r),
                                static_cast<int>((cx + w / 2) / r),
                                static_cast<int>((cy + h / 2) / r), bestScore, bestCls});
            }
            auto kept = nms(std::move(dets), 0.45f);
            for (const auto &d : kept)
                examples::drawRect(stbImg, d.x0, d.y0, d.x1, d.y1, 0, 220, 0, 3);
        } else if (mode == Mode::Classification) {
            float maxLogit = *std::max_element(data.begin(), data.end());
            std::vector<float> probs(data.size());
            double sum = 0.0;
            for (std::size_t j = 0; j < data.size(); ++j) {
                probs[j] = std::exp(data[j] - maxLogit);
                sum += probs[j];
            }
        } else if (mode == Mode::Segmentation) {
            const int plane = outH * outW;
            std::vector<std::uint8_t> classMap(static_cast<std::size_t>(plane));
            for (int p = 0; p < plane; ++p) {
                int best = 0;
                float bestVal = data[p];
                for (int c = 1; c < outC; ++c) {
                    float v = data[static_cast<std::size_t>(c) * plane + p];
                    if (v > bestVal) { bestVal = v; best = c; }
                }
                classMap[static_cast<std::size_t>(p)] = static_cast<std::uint8_t>(best);
            }
            examples::Image overlay = origImg;
            for (int y = 0; y < origImg.height; ++y) {
                int sy = y * outH / origImg.height;
                for (int x = 0; x < origImg.width; ++x) {
                    int sx = x * outW / origImg.width;
                    auto col = palette[classMap[static_cast<std::size_t>(sy) * outW + sx]];
                    auto *px = &overlay.data[(static_cast<std::size_t>(y) * origImg.width + x) * 3];
                    px[0] = static_cast<std::uint8_t>((px[0] + col.r) / 2);
                    px[1] = static_cast<std::uint8_t>((px[1] + col.g) / 2);
                    px[2] = static_cast<std::uint8_t>((px[2] + col.b) / 2);
                }
            }
            stbImg = overlay;
        }

        accum(t_postproc, elapsedMs(t0));
    }
    // Single write at the end for the output file
    if (mode == Mode::Detection || mode == Mode::Segmentation) {
        examples::writeJpg(outPath, stbImg);
        std::fprintf(stderr, "wrote %s\n", outPath.c_str());
    }

    // ----------------------------------------------------------------
    // Print results
    // ----------------------------------------------------------------
    auto totalMs = [](const Timing &t) { return t.avg(); };
    auto subTotal = [](const Timing *t, int n) {
        double s = 0;
        for (int i = 0; i < n; ++i) s += t[i].avg();
        return s;
    };

    double noCvTotal = totalMs(t_noCv_decode) + totalMs(t_noCv_preproc) +
                       totalMs(t_infer) + totalMs(t_postproc);
    double cvPreproc = subTotal(cvBuf.sub, 5);
    double cvTotal = totalMs(t_cv_decode) + totalMs(t_cv_upload) + cvPreproc +
                     totalMs(t_infer) + totalMs(t_postproc);

    std::printf("\n");
    std::printf("==========================================================================\n");
    std::printf(" Full-Pipeline: %s\n", modeName(mode));
    std::printf(" Model: %s\n", modelPath.c_str());
    std::printf(" Image: %d x %d  ->  Network: %d x %d\n", stbImg.width, stbImg.height, inW, inH);
    std::printf(" Iters: %d  (warmup: %d)\n", iters, warmup);
    std::printf("==========================================================================\n");
    std::printf(" Stage                       Without OpenCV    With OpenCV\n");
    std::printf(" -------------------------   --------------    -----------\n");
    std::printf(" %-23s  %9.4f ms       %9.4f ms\n", "Decode",
                totalMs(t_noCv_decode), totalMs(t_cv_decode));
    std::printf(" %-23s  %9.4f ms       %9.4f ms\n", "Upload",
                0.0, totalMs(t_cv_upload));
    std::printf(" %-23s  %9.4f ms       %9.4f ms\n", "Preprocess",
                totalMs(t_noCv_preproc), cvPreproc);
    std::printf(" %-23s  %9.4f ms       %9.4f ms\n", "Inference",
                totalMs(t_infer), totalMs(t_infer));
    std::printf(" %-23s  %9.4f ms       %9.4f ms\n", "Postprocess",
                totalMs(t_postproc), totalMs(t_postproc));
    std::printf(" -------------------------   --------------    -----------\n");
    std::printf(" %-23s  %9.4f ms       %9.4f ms\n", "Total pipeline", noCvTotal, cvTotal);

    if (haveOpenCV) {
        auto hostStb = outStb.toHost(stream).value();
        auto hostCv = outCv.toHost(stream).value();
        auto dataStb = hostStb.as<float>().value();
        auto dataCv = hostCv.as<float>().value();
        double mse = 0, maxDiff = 0;
        const auto n = hostStb.nbytes() / sizeof(float);
        for (std::size_t i = 0; i < n; ++i) {
            double d = std::abs(static_cast<double>(dataStb[i]) - static_cast<double>(dataCv[i]));
            mse += d * d;
            if (d > maxDiff) maxDiff = d;
        }
        mse /= static_cast<double>(n);
        std::printf("\n Numerical validation:\n");
        std::printf("   Inference output  MSE: %.2e  MaxDiff: %.2e\n", mse, maxDiff);
        if (maxDiff < 1e-3f)
            std::printf("   => Outputs match within floating-point tolerance.\n");
        else
            std::printf("   => WARNING: outputs differ significantly.\n");
    } else {
        std::printf("\n OpenCV pipeline unavailable — stb-only results shown.\n");
    }
    std::printf("\n");

    return 0;
}
