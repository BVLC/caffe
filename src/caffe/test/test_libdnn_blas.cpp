#ifdef USE_LIBDNN

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/libdnn/libdnn_blas.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class GemmTest : public ::testing::Test {};

TYPED_TEST_CASE(LibDNNGemmTest, TestDtypesFloat);

TYPED_TEST(LibDNNGemmTest, TestGemmCPUGPU) {
  Device *dc = Caffe::GetDefaultDevice();

  Blob<TypeParam> A(1, 1, 2, 3, Caffe::GetDefaultDevice());
  Blob<TypeParam> B(1, 1, 3, 4, Caffe::GetDefaultDevice());
  Blob<TypeParam> C(1, 1, 2, 4, Caffe::GetDefaultDevice());
  TypeParam data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TypeParam A_reshape_data[6] = {1, 4, 2, 5, 3, 6};
  TypeParam B_reshape_data[12] = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  TypeParam result[8] = {38, 44, 50, 56, 83, 98, 113, 128};

  caffe_cpu_copy(6, data, A.mutable_cpu_data());
  caffe_cpu_copy(12, data, B.mutable_cpu_data());

  // [1, 2, 3; 4 5 6] * [1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
  caffe_cpu_gemm<TypeParam>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
      A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  dc->get_libdnn_blas<float,float,float>()->gemm(CblasNoTrans, CblasNoTrans,
              2, 4, 3, 1., A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data(),
              LIBDNN_ACCUMULATE_PREC_NATIVE,
              std::make_shared<Quantizer<float, float> >(dc)
              std::make_shared<Quantizer<float, float> >(dc));

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  // Test when we have a transposed A
  A.Reshape(1, 1, 3, 2);
  caffe_cpu_copy(6, A_reshape_data, A.mutable_cpu_data());
  caffe_cpu_gemm<TypeParam>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
      A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  dc->get_libdnn_blas<float,float,float>()->gemm(CblasNoTrans, CblasNoTrans,
              2, 4, 3, 1., A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data(),
              LIBDNN_ACCUMULATE_PREC_NATIVE,
              std::make_shared<Quantizer<float, float> >(dc)
              std::make_shared<Quantizer<float, float> >(dc));

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  // Test when we have a transposed A and a transposed B too
  B.Reshape(1, 1, 4, 3);
  caffe_cpu_copy(12, B_reshape_data, B.mutable_cpu_data());
  caffe_cpu_gemm<TypeParam>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
      A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  dc->get_libdnn_blas<float,float,float>()->gemm(CblasNoTrans, CblasNoTrans,
              2, 4, 3, 1., A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data(),
              LIBDNN_ACCUMULATE_PREC_NATIVE,
              std::make_shared<Quantizer<float, float> >(dc)
              std::make_shared<Quantizer<float, float> >(dc));

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  // Test when we have a transposed B
  A.Reshape(1, 1, 2, 3);
  caffe_cpu_copy(6, data, A.mutable_cpu_data());
  caffe_cpu_gemm<TypeParam>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
      A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  dc->get_libdnn_blas<float,float,float>()->gemm(CblasNoTrans, CblasNoTrans,
              2, 4, 3, 1., A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data(),
              LIBDNN_ACCUMULATE_PREC_NATIVE,
              std::make_shared<Quantizer<float, float> >(dc)
              std::make_shared<Quantizer<float, float> >(dc));

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }
}



#endif  // USE_LIBDNN
