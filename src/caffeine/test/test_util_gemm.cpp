#include <cstring>
#include <cuda_runtime.h>
#include <mkl.h>
#include <cublas_v2.h>

#include "gtest/gtest.h"
#include "caffeine/blob.hpp"
#include "caffeine/util/gemm.hpp"

namespace caffeine {

extern cudaDeviceProp CAFFEINE_TEST_CUDA_PROP;

typedef ::testing::Types<float, double> Dtypes;

template <typename Dtype>
class GemmTest : public ::testing::Test {};

TYPED_TEST_CASE(GemmTest, Dtypes);

TYPED_TEST(GemmTest, TestGemm) {
  Blob<TypeParam> A(1,1,2,3);
  Blob<TypeParam> B(1,1,3,4);
  Blob<TypeParam> C(1,1,2,4);
  TypeParam data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TypeParam A_reshape_data[6] = {1, 4, 2, 5, 3, 6};
  TypeParam B_reshape_data[12] = {1,5,9,2,6,10,3,7,11,4,8,12};
  TypeParam result[8] = {38,44,50,56,83,98,113,128};
  memcpy(A.mutable_cpu_data(), data, 6 * sizeof(TypeParam));
  memcpy(B.mutable_cpu_data(), data, 12 * sizeof(TypeParam));

  if (sizeof(TypeParam) == 4 || CAFFEINE_TEST_CUDA_PROP.major >= 2) {
    //[1,2,3; 4 5 6] * [1,2,3,4; 5,6,7,8; 9,10,11,12];
    decaf_cpu_gemm<TypeParam>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    decaf_gpu_gemm<TypeParam>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }

    // Test when we have a transposed A
    A.Reshape(1,1,3,2);
    memcpy(A.mutable_cpu_data(), A_reshape_data, 6 * sizeof(TypeParam));
    decaf_cpu_gemm<TypeParam>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    decaf_gpu_gemm<TypeParam>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }

    // Test when we have a transposed A and a transposed B too
    B.Reshape(1,1,4,3);
    memcpy(B.mutable_cpu_data(), B_reshape_data, 12 * sizeof(TypeParam));
    decaf_cpu_gemm<TypeParam>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    decaf_gpu_gemm<TypeParam>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }

    // Test when we have a transposed B
    A.Reshape(1,1,2,3);
    memcpy(A.mutable_cpu_data(), data, 6 * sizeof(TypeParam));
    decaf_cpu_gemm<TypeParam>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    decaf_gpu_gemm<TypeParam>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


}
