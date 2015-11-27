#ifndef CPU_ONLY  // CPU-GPU test

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename TypeParam>
class GemmTest : public ::testing::Test {};

TYPED_TEST_CASE(GemmTest, TestDtypes);

TYPED_TEST(GemmTest, TestGemmCPUGPU) {
  Blob<TypeParam> A(1, 1, 2, 3);
  Blob<TypeParam> B(1, 1, 3, 4);
  Blob<TypeParam> C(1, 1, 2, 4);
  TypeParam data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TypeParam A_reshape_data[6] = {1, 4, 2, 5, 3, 6};
  TypeParam B_reshape_data[12] = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  TypeParam result[8] = {38, 44, 50, 56, 83, 98, 113, 128};
  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_copy(12, data, B.mutable_cpu_data());

  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    // [1, 2, 3; 4 5 6] * [1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
    caffe_cpu_gemm<TypeParam>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    caffe_gpu_gemm<TypeParam>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }

    // Test when we have a transposed A
    A.Reshape(1, 1, 3, 2);
    caffe_copy(6, A_reshape_data, A.mutable_cpu_data());
    caffe_cpu_gemm<TypeParam>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    caffe_gpu_gemm<TypeParam>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }

    // Test when we have a transposed A and a transposed B too
    B.Reshape(1, 1, 4, 3);
    caffe_copy(12, B_reshape_data, B.mutable_cpu_data());
    caffe_cpu_gemm<TypeParam>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    caffe_gpu_gemm<TypeParam>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }

    // Test when we have a transposed B
    A.Reshape(1, 1, 2, 3);
    caffe_copy(6, data, A.mutable_cpu_data());
    caffe_cpu_gemm<TypeParam>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    caffe_gpu_gemm<TypeParam>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


TYPED_TEST(GemmTest, TestGemvCPUGPU) {
  Blob<TypeParam> A(1, 1, 2, 3);
  Blob<TypeParam> x(1, 1, 1, 3);
  Blob<TypeParam> y(1, 1, 1, 2);
  TypeParam data[6] = {1, 2, 3, 4, 5, 6};
  TypeParam result_2[2] = {14, 32};
  TypeParam result_3[3] = {9, 12, 15};
  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_copy(3, data, x.mutable_cpu_data());

  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    caffe_cpu_gemv<TypeParam>(CblasNoTrans, 2, 3, 1., A.cpu_data(),
        x.cpu_data(), 0., y.mutable_cpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(y.cpu_data()[i], result_2[i]);
    }
    caffe_gpu_gemv<TypeParam>(CblasNoTrans, 2, 3, 1., A.gpu_data(),
        x.gpu_data(), 0., y.mutable_gpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(y.cpu_data()[i], result_2[i]);
    }

    // Test transpose case
    caffe_copy(2, data, y.mutable_cpu_data());
    caffe_cpu_gemv<TypeParam>(CblasTrans, 2, 3, 1., A.cpu_data(),
        y.cpu_data(), 0., x.mutable_cpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(x.cpu_data()[i], result_3[i]);
    }
    caffe_gpu_gemv<TypeParam>(CblasTrans, 2, 3, 1., A.gpu_data(),
        y.gpu_data(), 0., x.mutable_gpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(x.cpu_data()[i], result_3[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe

#endif  // CPU_ONLY
