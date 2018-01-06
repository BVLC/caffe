#ifndef CPU_ONLY  // CPU-GPU test

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class GemmTest : public ::testing::Test {};

TYPED_TEST_CASE(GemmTest, TestDtypesFloat);

TYPED_TEST(GemmTest, TestGemmCPUGPU) {
  Device *dc = Caffe::GetDefaultDevice();

  Blob<TypeParam> a(1, 1, 2, 3, Caffe::GetDefaultDevice());
  Blob<TypeParam> b(1, 1, 3, 4, Caffe::GetDefaultDevice());
  Blob<TypeParam> c(1, 1, 2, 4, Caffe::GetDefaultDevice());
  TypeParam data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TypeParam A_reshape_data[6] = {1, 4, 2, 5, 3, 6};
  TypeParam B_reshape_data[12] = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  TypeParam result[8] = {38, 44, 50, 56, 83, 98, 113, 128};

  caffe_copy(6, data, a.mutable_cpu_data());
  caffe_copy(12, data, b.mutable_cpu_data());

  // [1, 2, 3; 4 5 6] * [1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
  caffe_gemm<TypeParam>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
      a.cpu_data(), b.cpu_data(), 0., c.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(c.cpu_data()[i], result[i]);
  }

  dc->gemm<TypeParam>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
      a.gpu_data(), b.gpu_data(), 0., c.mutable_gpu_data());

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(c.cpu_data()[i], result[i]);
  }

  // Test when we have a transposed a
  a.Reshape(1, 1, 3, 2);
  caffe_copy(6, A_reshape_data, a.mutable_cpu_data());
  caffe_gemm<TypeParam>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
      a.cpu_data(), b.cpu_data(), 0., c.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(c.cpu_data()[i], result[i]);
  }

  dc->gemm<TypeParam>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
                      a.gpu_data(), b.gpu_data(), 0., c.mutable_gpu_data());

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(c.cpu_data()[i], result[i]);
  }

  // Test when we have a transposed a and a transposed b too
  b.Reshape(1, 1, 4, 3);
  caffe_copy(12, B_reshape_data, b.mutable_cpu_data());
  caffe_gemm<TypeParam>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
      a.cpu_data(), b.cpu_data(), 0., c.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(c.cpu_data()[i], result[i]);
  }

  dc->gemm<TypeParam>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
                      a.gpu_data(), b.gpu_data(), 0., c.mutable_gpu_data());

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(c.cpu_data()[i], result[i]);
  }

  // Test when we have a transposed b
  a.Reshape(1, 1, 2, 3);
  caffe_copy(6, data, a.mutable_cpu_data());
  caffe_gemm<TypeParam>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
      a.cpu_data(), b.cpu_data(), 0., c.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(c.cpu_data()[i], result[i]);
  }

  dc->gemm<TypeParam>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
      a.gpu_data(), b.gpu_data(), 0., c.mutable_gpu_data());

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(c.cpu_data()[i], result[i]);
  }
}


TYPED_TEST(GemmTest, TestGemvCPUGPU) {
  Device *dc = Caffe::GetDefaultDevice();

  Blob<TypeParam> a(1, 1, 2, 3, Caffe::GetDefaultDevice());
  Blob<TypeParam> X(1, 1, 1, 3, Caffe::GetDefaultDevice());
  Blob<TypeParam> Y(1, 1, 1, 2, Caffe::GetDefaultDevice());
  TypeParam data[6] = {1, 2, 3, 4, 5, 6};
  TypeParam result_2[2] = {14, 32};
  TypeParam result_3[3] = {9, 12, 15};

  caffe_copy(6, data, a.mutable_cpu_data());
  caffe_copy(3, data, X.mutable_cpu_data());

  caffe_gemv<TypeParam>(CblasNoTrans, 2, 3, 1., a.cpu_data(),
      X.cpu_data(), 0., Y.mutable_cpu_data());

  for (int_tp i = 0; i < 2; ++i) {
    EXPECT_EQ(Y.cpu_data()[i], result_2[i]);
  }

  dc->gemv<TypeParam>(CblasNoTrans, 2, 3, 1., a.gpu_data(),
                      X.gpu_data(), 0., Y.mutable_gpu_data());

  for (int_tp i = 0; i < 2; ++i) {
    EXPECT_EQ(Y.cpu_data()[i], result_2[i]);
  }

  // Test transpose case
  caffe_copy(2, data, Y.mutable_cpu_data());
  caffe_gemv<TypeParam>(CblasTrans, 2, 3, 1., a.cpu_data(),
      Y.cpu_data(), 0., X.mutable_cpu_data());
  for (int_tp i = 0; i < 3; ++i) {
    EXPECT_EQ(X.cpu_data()[i], result_3[i]);
  }

  dc->gemv<TypeParam>(CblasTrans, 2, 3, 1., a.gpu_data(),
                      Y.gpu_data(), 0., X.mutable_gpu_data());

  for (int_tp i = 0; i < 3; ++i) {
    EXPECT_EQ(X.cpu_data()[i], result_3[i]);
  }
}

}  // namespace caffe

#endif  // CPU_ONLY
