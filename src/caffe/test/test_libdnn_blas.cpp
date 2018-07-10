#ifdef USE_LIBDNN

#include <algorithm>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/libdnn/libdnn_blas.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

// Comparative check difference limit
#define KAPPA_HALF 0.05
#define KAPPA_FLOAT 0.05
#define KAPPA_DOUBLE 0.05

#define EPS_HALF 5e-1
#define EPS_FLOAT 1e-4
#define EPS_DOUBLE 1e-4

namespace caffe {

template <typename TypeParam>
class LibDNNBlasTest : public ::testing::Test {};

TYPED_TEST_CASE(LibDNNBlasTest, TestDtypesFloat);

TYPED_TEST(LibDNNBlasTest, TestGemmCPUGPU) {
  Device *dc = Caffe::GetDefaultDevice();

  Blob<TypeParam> A(1, 1, 2, 3, Caffe::GetDefaultDevice());
  Blob<TypeParam> B(1, 1, 3, 4, Caffe::GetDefaultDevice());
  Blob<TypeParam> C(1, 1, 2, 4, Caffe::GetDefaultDevice());
  TypeParam data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  TypeParam A_reshape_data[6] = {1, 4, 2, 5, 3, 6};
  TypeParam B_reshape_data[12] = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  TypeParam result[8] = {38, 44, 50, 56, 83, 98, 113, 128};

  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_copy(12, data, B.mutable_cpu_data());

  // [1, 2, 3; 4 5 6] * [1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
  caffe_gemm<TypeParam>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
      A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  dc->GetLibDNNBlas<TypeParam, TypeParam>()->gemm(CblasNoTrans, CblasNoTrans,
              2, 4, 3, 1., A.gpu_data(), B.gpu_data(), 0.,
              C.mutable_gpu_data());

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  // Test when we have a transposed A
  A.Reshape(1, 1, 3, 2);
  caffe_copy(6, A_reshape_data, A.mutable_cpu_data());
  caffe_gemm<TypeParam>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
      A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  dc->GetLibDNNBlas<TypeParam, TypeParam>()->gemm(CblasTrans, CblasNoTrans,
              2, 4, 3, 1., A.gpu_data(), B.gpu_data(), 0.,
              C.mutable_gpu_data());

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  // Test when we have a transposed A and a transposed B too
  B.Reshape(1, 1, 4, 3);
  caffe_copy(12, B_reshape_data, B.mutable_cpu_data());
  caffe_gemm<TypeParam>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
      A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  dc->GetLibDNNBlas<TypeParam, TypeParam>()->gemm(CblasTrans, CblasTrans,
              2, 4, 3, 1., A.gpu_data(), B.gpu_data(), 0.,
              C.mutable_gpu_data());

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  // Test when we have a transposed B
  A.Reshape(1, 1, 2, 3);
  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_gemm<TypeParam>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
      A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }

  dc->GetLibDNNBlas<TypeParam, TypeParam>()->gemm(CblasNoTrans, CblasTrans,
              2, 4, 3, 1., A.gpu_data(), B.gpu_data(), 0.,
              C.mutable_gpu_data());

  for (int_tp i = 0; i < 8; ++i) {
    EXPECT_EQ(C.cpu_data()[i], result[i]);
  }
}


TYPED_TEST(LibDNNBlasTest, TestGemmComparativeCPUGPU) {
  Device *dc = Caffe::GetDefaultDevice();

  TypeParam eps = 0.0;
  if (std::is_same<TypeParam, half_fp>::value) {
    eps = EPS_HALF;
  }
  if (std::is_same<TypeParam, float>::value) {
    eps = EPS_FLOAT;
  }
  if (std::is_same<TypeParam, double>::value) {
    eps = EPS_DOUBLE;
  }

  std::random_device rdev;
  std::mt19937 rngen(rdev());

  std::uniform_int_distribution<int_tp> dimsRand(1, 256);
  std::uniform_int_distribution<int_tp> boolRand(0, 1);
  std::uniform_int_distribution<int_tp> factorRand(-25, 25);

  for (int_tp testIdx = 0; testIdx < 25; ++testIdx) {
    int_tp M = dimsRand(rngen);
    int_tp N = dimsRand(rngen);
    int_tp K = dimsRand(rngen);

    CBLAS_TRANSPOSE trans_A = boolRand(rngen) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_B = boolRand(rngen) ? CblasTrans : CblasNoTrans;

    bool has_alpha = boolRand(rngen);
    TypeParam alpha_val = factorRand(rngen) / 100.0;
    bool has_beta = boolRand(rngen);
    TypeParam beta_val = factorRand(rngen) / 100.0;

    vector<int_tp> A_shape(4, 1);
    vector<int_tp> B_shape(4, 1);
    vector<int_tp> C_shape(4, 1);

    A_shape[2] = M;
    A_shape[3] = K;
    B_shape[2] = K;
    B_shape[3] = N;
    C_shape[2] = M;
    C_shape[3] = N;

    Blob<TypeParam> A(A_shape, Caffe::GetDefaultDevice());
    Blob<TypeParam> B(B_shape, Caffe::GetDefaultDevice());
    Blob<TypeParam> C_GPU(C_shape, Caffe::GetDefaultDevice());
    Blob<TypeParam> C_CPU(C_shape, Caffe::GetDefaultDevice());

    caffe_rng_gaussian(M * K, (TypeParam)0.0, (TypeParam)0.25,
                       A.mutable_cpu_data());
    caffe_rng_gaussian(K * N, (TypeParam)0.0, (TypeParam)0.25,
                       B.mutable_cpu_data());
    caffe_rng_gaussian(M * N, (TypeParam)0.0, (TypeParam)0.25,
                       C_CPU.mutable_cpu_data());
    caffe_copy(M * N, C_CPU.cpu_data(), C_GPU.mutable_cpu_data());

    std::cout << "==== Test Case " << testIdx << " ====" << std::endl;
    std::cout << "M: " << M << " N: " << N << " K: " << K << std::endl;
    std::cout << "alpha: " << (has_alpha ? alpha_val : (TypeParam)1.0) << " "
              << "beta: " << (has_beta ? beta_val : (TypeParam)0.0)
              << std::endl;
    std::cout << "trans A: " << (trans_A == CblasTrans) << " "
              << "trans B: " << (trans_B == CblasTrans) << std::endl;

    dc->GetLibDNNBlas<TypeParam, TypeParam>()->gemm(
                trans_A, trans_B,
                M, N, K,
                has_alpha ? alpha_val: (TypeParam)1.,
                A.gpu_data(), B.gpu_data(),
                has_beta ? beta_val : (TypeParam)0.,
                C_GPU.mutable_gpu_data());

    caffe_gemm<TypeParam>(
                trans_A, trans_B,
                M, N, K,
                has_alpha ? alpha_val: (TypeParam)1.,
                A.cpu_data(), B.cpu_data(),
                has_beta ? beta_val : (TypeParam)0.,
                C_CPU.mutable_cpu_data());

    for (int_tp i = 0; i < M * N; ++i) {
      EXPECT_NEAR(C_CPU.cpu_data()[i], C_GPU.cpu_data()[i], eps);
      // One error is enough to abort
      if (fabs(C_CPU.cpu_data()[i] - C_GPU.cpu_data()[i]) >= eps) {
        break;
      }
    }
  }
}


TYPED_TEST(LibDNNBlasTest, TestGemvCPUGPU) {
  Device *dc = Caffe::GetDefaultDevice();

  Blob<TypeParam> A(1, 1, 2, 3, Caffe::GetDefaultDevice());
  Blob<TypeParam> x(1, 1, 1, 3, Caffe::GetDefaultDevice());
  Blob<TypeParam> y(1, 1, 1, 2, Caffe::GetDefaultDevice());
  TypeParam data[6] = {1, 2, 3, 4, 5, 6};
  TypeParam result_2[2] = {14, 32};
  TypeParam result_3[3] = {9, 12, 15};

  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_copy(3, data, x.mutable_cpu_data());


  caffe_gemv<TypeParam>(CblasNoTrans, 2, 3, 1., A.cpu_data(),
      x.cpu_data(), 0., y.mutable_cpu_data());
  for (int_tp i = 0; i < 2; ++i) {
    EXPECT_EQ(y.cpu_data()[i], result_2[i]);
  }

  dc->GetLibDNNBlas<TypeParam, TypeParam>()->gemv(CblasNoTrans,
                      2, 3, 1., A.gpu_data(),
                      x.gpu_data(), 0., y.mutable_gpu_data());

  for (int_tp i = 0; i < 2; ++i) {
    EXPECT_EQ(y.cpu_data()[i], result_2[i]);
  }

  // Test transpose case
  caffe_copy(2, data, y.mutable_cpu_data());
  caffe_gemv<TypeParam>(CblasTrans, 2, 3, 1., A.cpu_data(),
      y.cpu_data(), 0., x.mutable_cpu_data());
  for (int_tp i = 0; i < 3; ++i) {
    EXPECT_EQ(x.cpu_data()[i], result_3[i]);
  }

  dc->GetLibDNNBlas<TypeParam, TypeParam>()->gemv(CblasTrans,
                      2, 3, 1., A.gpu_data(),
                      y.gpu_data(), 0., x.mutable_gpu_data());

  for (int_tp i = 0; i < 3; ++i) {
    EXPECT_EQ(x.cpu_data()[i], result_3[i]);
  }
}


TYPED_TEST(LibDNNBlasTest, TestGemvComparativeCPUGPU) {
  Device *dc = Caffe::GetDefaultDevice();

  TypeParam eps = 0.0;
  if (std::is_same<TypeParam, half_fp>::value) {
    eps = EPS_HALF;
  }
  if (std::is_same<TypeParam, float>::value) {
    eps = EPS_FLOAT;
  }
  if (std::is_same<TypeParam, double>::value) {
    eps = EPS_DOUBLE;
  }

  std::random_device rdev;
  std::mt19937 rngen(rdev());

  std::uniform_int_distribution<int_tp> dimsRand(1, 256);
  std::uniform_int_distribution<int_tp> boolRand(0, 1);
  std::uniform_int_distribution<int_tp> factorRand(-25, 25);

  for (int_tp testIdx = 0; testIdx < 25; ++testIdx) {
    int_tp M = dimsRand(rngen);
    int_tp N = dimsRand(rngen);

    CBLAS_TRANSPOSE trans_A = boolRand(rngen) ? CblasTrans : CblasNoTrans;

    bool has_alpha = boolRand(rngen);
    TypeParam alpha_val = factorRand(rngen) / 100.0;
    bool has_beta = boolRand(rngen);
    TypeParam beta_val = factorRand(rngen) / 100.0;

    vector<int_tp> A_shape(4, 1);
    vector<int_tp> x_shape(4, 1);
    vector<int_tp> y_shape(4, 1);

    A_shape[2] = M;
    A_shape[3] = N;
    x_shape[3] = trans_A == CblasTrans ? M : N;
    y_shape[3] = trans_A == CblasTrans ? N : M;

    Blob<TypeParam> A(A_shape, Caffe::GetDefaultDevice());
    Blob<TypeParam> x(x_shape, Caffe::GetDefaultDevice());
    Blob<TypeParam> y_GPU(y_shape, Caffe::GetDefaultDevice());
    Blob<TypeParam> y_CPU(y_shape, Caffe::GetDefaultDevice());

    caffe_rng_gaussian(M * N, (TypeParam)0.0, (TypeParam)0.25,
                       A.mutable_cpu_data());
    caffe_rng_gaussian(trans_A == CblasTrans ? M : N, (TypeParam)0.0,
                       (TypeParam)0.25, x.mutable_cpu_data());
    caffe_rng_gaussian(trans_A == CblasTrans ? N : M, (TypeParam)0.0,
                       (TypeParam)0.25, y_CPU.mutable_cpu_data());
    caffe_copy(trans_A == CblasTrans ? N : M, y_CPU.cpu_data(),
               y_GPU.mutable_cpu_data());

    std::cout << "==== Test Case " << testIdx << " ====" << std::endl;
    std::cout << "M: " << M << " N: " << N << std::endl;
    std::cout << "alpha: " << (has_alpha ? alpha_val : (TypeParam)1.0) << " "
              << "beta: " << (has_beta ? beta_val : (TypeParam)0.0)
              << std::endl;
    std::cout << "trans A: " << (trans_A == CblasTrans) << std::endl;

    dc->GetLibDNNBlas<TypeParam, TypeParam>()->gemv(
                trans_A,
                M, N,
                has_alpha ? alpha_val: (TypeParam)1.,
                A.gpu_data(), x.gpu_data(),
                has_beta ? beta_val : (TypeParam)0.,
                y_GPU.mutable_gpu_data());

    caffe_gemv<TypeParam>(
                trans_A,
                M, N,
                has_alpha ? alpha_val: (TypeParam)1.,
                A.cpu_data(), x.cpu_data(),
                has_beta ? beta_val : (TypeParam)0.,
                y_CPU.mutable_cpu_data());

    for (int_tp i = 0; i < (trans_A == CblasTrans ? N : M); ++i) {
      EXPECT_NEAR(y_CPU.cpu_data()[i], y_GPU.cpu_data()[i], eps);
      // One error is enough to abort
      if (fabs(y_CPU.cpu_data()[i] - y_GPU.cpu_data()[i]) >= eps) {
        break;
      }
    }
  }
}


} // namespace caffe

#endif  // USE_LIBDNN
