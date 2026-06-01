#include "tensorrt_cpp_api/opencv_interop.h"

#ifdef TRT_CPP_API_WITH_OPENCV

#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

using namespace trtcpp;

// ------------------------------------------------------------------
// dtypeOfCvDepth
// ------------------------------------------------------------------

TEST(OpenCVInterop, DtypeOfCvDepthU8) {
    auto dt = opencv::dtypeOfCvDepth(CV_8U);
    ASSERT_TRUE(dt.ok());
    EXPECT_EQ(dt.value(), DType::kUInt8);
}

TEST(OpenCVInterop, DtypeOfCvDepthS8) {
    auto dt = opencv::dtypeOfCvDepth(CV_8S);
    ASSERT_TRUE(dt.ok());
    EXPECT_EQ(dt.value(), DType::kInt8);
}

TEST(OpenCVInterop, DtypeOfCvDepthS32) {
    auto dt = opencv::dtypeOfCvDepth(CV_32S);
    ASSERT_TRUE(dt.ok());
    EXPECT_EQ(dt.value(), DType::kInt32);
}

TEST(OpenCVInterop, DtypeOfCvDepthF32) {
    auto dt = opencv::dtypeOfCvDepth(CV_32F);
    ASSERT_TRUE(dt.ok());
    EXPECT_EQ(dt.value(), DType::kFloat32);
}

TEST(OpenCVInterop, DtypeOfCvDepthF16) {
    auto dt = opencv::dtypeOfCvDepth(CV_16F);
    ASSERT_TRUE(dt.ok());
    EXPECT_EQ(dt.value(), DType::kFloat16);
}

TEST(OpenCVInterop, DtypeOfCvDepth64FRejected) {
    auto dt = opencv::dtypeOfCvDepth(CV_64F);
    EXPECT_FALSE(dt.ok());
    EXPECT_EQ(dt.status().code(), StatusCode::kUnsupported);
}

TEST(OpenCVInterop, DtypeOfCvDepth16URejected) {
    auto dt = opencv::dtypeOfCvDepth(CV_16U);
    EXPECT_FALSE(dt.ok());
    EXPECT_EQ(dt.status().code(), StatusCode::kUnsupported);
}

// ------------------------------------------------------------------
// viewOf(cv::Mat)
// ------------------------------------------------------------------

TEST(OpenCVInterop, MatHostViewIsHwc) {
    cv::Mat mat(4, 5, CV_8UC3, cv::Scalar(1, 2, 3));
    auto view = opencv::viewOf(mat);
    ASSERT_TRUE(view.ok()) << view.status().message();
    EXPECT_EQ(view.value().dtype(), DType::kUInt8);
    EXPECT_EQ(view.value().shape(), (Shape{4, 5, 3}));
    EXPECT_EQ(view.value().device(), Device::kHost);
    EXPECT_EQ(view.value().data(), mat.data);
}

TEST(OpenCVInterop, MatHostViewSingleChannel) {
    cv::Mat mat(3, 7, CV_32FC1, cv::Scalar(0.0f));
    auto view = opencv::viewOf(mat);
    ASSERT_TRUE(view.ok());
    EXPECT_EQ(view.value().shape(), (Shape{3, 7, 1}));
    EXPECT_EQ(view.value().dtype(), DType::kFloat32);
}

TEST(OpenCVInterop, MatHostViewFourChannel) {
    cv::Mat mat(2, 3, CV_8UC4, cv::Scalar(10, 20, 30, 255));
    auto view = opencv::viewOf(mat);
    ASSERT_TRUE(view.ok());
    EXPECT_EQ(view.value().shape(), (Shape{2, 3, 4}));
    EXPECT_EQ(view.value().dtype(), DType::kUInt8);
}

TEST(OpenCVInterop, MatHostViewAllSupportedDepths) {
    struct { int cvDepth; DType expected; } cases[] = {
        {CV_8U,  DType::kUInt8},
        {CV_8S,  DType::kInt8},
        {CV_32S, DType::kInt32},
        {CV_32F, DType::kFloat32},
        {CV_16F, DType::kFloat16},
    };
    for (const auto &c : cases) {
        cv::Mat mat(2, 2, CV_MAKETYPE(c.cvDepth, 1));
        auto view = opencv::viewOf(mat);
        EXPECT_TRUE(view.ok()) << "cvDepth=" << c.cvDepth;
        if (view.ok()) {
            EXPECT_EQ(view.value().dtype(), c.expected);
        }
    }
}

TEST(OpenCVInterop, MatHostViewNonContinuousRejected) {
    cv::Mat mat(4, 4, CV_32FC1);
    cv::Mat roi = mat(cv::Rect(1, 1, 2, 2)); // non-continuous ROI
    EXPECT_FALSE(roi.isContinuous());
    auto view = opencv::viewOf(roi);
    EXPECT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), StatusCode::kUnsupported);
}

TEST(OpenCVInterop, MatHostViewEmptyRejected) {
    cv::Mat empty;
    auto view = opencv::viewOf(empty);
    EXPECT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), StatusCode::kInvalidArgument);
}

TEST(OpenCVInterop, UnsupportedDepthRejected) {
    cv::Mat mat(2, 2, CV_64FC1);
    auto view = opencv::viewOf(mat);
    EXPECT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), StatusCode::kUnsupported);
}

// ------------------------------------------------------------------
// upload + roundtrip
// ------------------------------------------------------------------

TEST(OpenCVInterop, UploadRoundtrip) {
    cv::Mat mat(2, 2, CV_32FC1);
    mat.at<float>(0, 0) = 1.0f;
    mat.at<float>(0, 1) = 2.0f;
    mat.at<float>(1, 0) = 3.0f;
    mat.at<float>(1, 1) = 4.0f;

    Stream stream;
    auto device = opencv::upload(mat, stream);
    ASSERT_TRUE(device.ok()) << device.status().message();
    EXPECT_EQ(device.value().device(), Device::kCuda);
    EXPECT_EQ(device.value().shape(), (Shape{2, 2, 1}));

    auto host = device.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<float>();
    ASSERT_TRUE(values.ok());
    ASSERT_EQ(values.value().size(), 4u);
    EXPECT_FLOAT_EQ(values.value()[0], 1.0f);
    EXPECT_FLOAT_EQ(values.value()[3], 4.0f);
}

TEST(OpenCVInterop, UploadMultiChannelRoundtrip) {
    cv::Mat mat(3, 3, CV_8UC3, cv::Scalar(10, 20, 30));
    Stream stream;
    auto device = opencv::upload(mat, stream);
    ASSERT_TRUE(device.ok());
    EXPECT_EQ(device.value().shape(), (Shape{3, 3, 3}));
    EXPECT_EQ(device.value().dtype(), DType::kUInt8);

    auto host = device.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto pixels = host.value().as<uint8_t>();
    ASSERT_TRUE(pixels.ok());
    EXPECT_EQ(pixels.value()[0], 10);  // R of (0,0)
    EXPECT_EQ(pixels.value()[1], 20);  // G of (0,0)
    EXPECT_EQ(pixels.value()[2], 30);  // B of (0,0)
}

TEST(OpenCVInterop, UploadInt8Roundtrip) {
    cv::Mat mat(1, 4, CV_8SC1);
    mat.at<int8_t>(0, 0) = -1;
    mat.at<int8_t>(0, 1) = 0;
    mat.at<int8_t>(0, 2) = 42;
    mat.at<int8_t>(0, 3) = 127;

    Stream stream;
    auto device = opencv::upload(mat, stream);
    ASSERT_TRUE(device.ok());
    auto host = device.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<int8_t>();
    ASSERT_TRUE(values.ok());
    EXPECT_EQ(values.value()[0], -1);
    EXPECT_EQ(values.value()[2], 42);
}

TEST(OpenCVInterop, UploadInt32Roundtrip) {
    cv::Mat mat(1, 3, CV_32SC1);
    mat.at<int32_t>(0, 0) = 100000;
    mat.at<int32_t>(0, 1) = -999;
    mat.at<int32_t>(0, 2) = 0;

    Stream stream;
    auto device = opencv::upload(mat, stream);
    ASSERT_TRUE(device.ok());
    auto host = device.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<int32_t>();
    ASSERT_TRUE(values.ok());
    EXPECT_EQ(values.value()[0], 100000);
    EXPECT_EQ(values.value()[1], -999);
}

TEST(OpenCVInterop, UploadEmptyRejected) {
    cv::Mat empty;
    Stream stream;
    auto device = opencv::upload(empty, stream);
    EXPECT_FALSE(device.ok());
    EXPECT_EQ(device.status().code(), StatusCode::kInvalidArgument);
}

// ------------------------------------------------------------------
// viewOf(cv::cuda::GpuMat)
// ------------------------------------------------------------------

TEST(OpenCVInterop, ContinuousGpuMatView) {
    cv::cuda::GpuMat gpu(4, 8, CV_8UC3);
    auto view = opencv::viewOf(gpu);
    if (view.ok()) {
        EXPECT_EQ(view.value().device(), Device::kCuda);
        EXPECT_EQ(view.value().shape(), (Shape{4, 8, 3}));
        EXPECT_EQ(view.value().dtype(), DType::kUInt8);
    } else {
        EXPECT_EQ(view.status().code(), StatusCode::kUnsupported);
    }
}

TEST(OpenCVInterop, GpuMatViewSingleChannel) {
    cv::cuda::GpuMat gpu(2, 5, CV_32FC1);
    auto view = opencv::viewOf(gpu);
    if (view.ok()) {
        EXPECT_EQ(view.value().shape(), (Shape{2, 5, 1}));
        EXPECT_EQ(view.value().dtype(), DType::kFloat32);
    } else {
        EXPECT_EQ(view.status().code(), StatusCode::kUnsupported);
    }
}

TEST(OpenCVInterop, GpuMatViewEmptyRejected) {
    cv::cuda::GpuMat empty;
    auto view = opencv::viewOf(empty);
    EXPECT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), StatusCode::kInvalidArgument);
}

// ------------------------------------------------------------------
// copyTo (pitched GpuMat -> continuous device Tensor)
// ------------------------------------------------------------------

TEST(OpenCVInterop, CopyFromPitchedGpuMat) {
    // Create a GpuMat that is likely pitched (step > cols*elemSize).
    // Creating with odd dimensions or using allocation from a GpuMat pool often results in pitch.
    cv::cuda::GpuMat gpu(4, 7, CV_32FC1);
    gpu.setTo(cv::Scalar(42.0f));
    // Verify pitch is present (if not, test still passes -- cudaMemcpy2D works either way).
    Stream stream;
    auto tensor = opencv::copyTo(gpu, stream);
    ASSERT_TRUE(tensor.ok()) << tensor.status().message();
    EXPECT_EQ(tensor.value().device(), Device::kCuda);
    EXPECT_EQ(tensor.value().shape(), (Shape{4, 7, 1}));

    auto host = tensor.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto data = host.value().as<float>();
    ASSERT_TRUE(data.ok());
    ASSERT_EQ(data.value().size(), 28u);
    for (std::size_t i = 0; i < data.value().size(); ++i) {
        EXPECT_FLOAT_EQ(data.value()[i], 42.0f);
    }
}

TEST(OpenCVInterop, CopyFromPitchedGpuMatMultiChannel) {
    cv::cuda::GpuMat gpu(3, 5, CV_8UC3);
    gpu.setTo(cv::Scalar(10, 20, 30));
    Stream stream;
    auto tensor = opencv::copyTo(gpu, stream);
    ASSERT_TRUE(tensor.ok());
    EXPECT_EQ(tensor.value().shape(), (Shape{3, 5, 3}));

    auto host = tensor.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto pixels = host.value().as<uint8_t>();
    ASSERT_TRUE(pixels.ok());
    EXPECT_EQ(pixels.value()[0], 10);
    EXPECT_EQ(pixels.value()[1], 20);
    EXPECT_EQ(pixels.value()[2], 30);
    EXPECT_EQ(pixels.value()[3], 10); // next pixel
}

TEST(OpenCVInterop, CopyFromGpuMatEmptyRejected) {
    cv::cuda::GpuMat empty;
    Stream stream;
    auto tensor = opencv::copyTo(empty, stream);
    EXPECT_FALSE(tensor.ok());
    EXPECT_EQ(tensor.status().code(), StatusCode::kInvalidArgument);
}

#endif // TRT_CPP_API_WITH_OPENCV
