#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
void caffe_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
                const int_tp M, const int_tp N, const int_tp K,
                const Dtype alpha, const Dtype* A,
                const Dtype* B, const Dtype beta, Dtype* C) {
  int_tp inc_a = (trans_A == CblasNoTrans) ? 1 : M;
  int_tp inc_b = (trans_B == CblasNoTrans) ? N : 1;
  for (int_tp m = 0; m < M; ++m) {
    for (int_tp n = 0; n < N; ++n) {
      Dtype acc = 0;
      int_tp b_index = trans_B == CblasNoTrans ? n : K * n;
      int_tp a_index = trans_A == CblasNoTrans ? K * m : m;
      for (int_tp k = 0; k < K; ++k) {
        acc += A[a_index] * B[b_index];
        a_index += inc_a;
        b_index += inc_b;
      }
      if (beta != 0) {
        C[m * N + n] = acc * alpha + beta * C[m * N + n];
      }
      else {
        C[m * N + n] = acc * alpha;
      }
    }
  }
}

template void caffe_gemm<half_fp>(const CBLAS_TRANSPOSE trans_A,
                             const CBLAS_TRANSPOSE trans_B,
                             const int_tp M, const int_tp N, const int_tp K,
                             const half_fp alpha, const half_fp* A,
                             const half_fp* B, const half_fp beta, half_fp* C);
template void caffe_gemm<int8_t>(const CBLAS_TRANSPOSE trans_A,
                             const CBLAS_TRANSPOSE trans_B,
                             const int_tp M, const int_tp N, const int_tp K,
                             const int8_t alpha, const int8_t* A,
                             const int8_t* B, const int8_t beta, int8_t* C);
template void caffe_gemm<int16_t>(const CBLAS_TRANSPOSE trans_A,
                             const CBLAS_TRANSPOSE trans_B,
                             const int_tp M, const int_tp N, const int_tp K,
                             const int16_t alpha, const int16_t* A,
                             const int16_t* B, const int16_t beta, int16_t* C);
template void caffe_gemm<int32_t>(const CBLAS_TRANSPOSE trans_A,
                             const CBLAS_TRANSPOSE trans_B,
                             const int_tp M, const int_tp N, const int_tp K,
                             const int32_t alpha, const int32_t* A,
                             const int32_t* B, const int32_t beta, int32_t* C);
template void caffe_gemm<int64_t>(const CBLAS_TRANSPOSE trans_A,
                             const CBLAS_TRANSPOSE trans_B,
                             const int_tp M, const int_tp N, const int_tp K,
                             const int64_t alpha, const int64_t* A,
                             const int64_t* B, const int64_t beta, int64_t* C);

template<>
void caffe_gemm<float>(const CBLAS_TRANSPOSE trans_A,
                           const CBLAS_TRANSPOSE trans_B, const int_tp M,
                           const int_tp N, const int_tp K, const float alpha,
                           const float* A, const float* B, const float beta,
                           float* C) {
  int_tp lda = (trans_A == CblasNoTrans) ? K : M;
  int_tp ldb = (trans_B == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

template<>
void caffe_gemm<double>(const CBLAS_TRANSPOSE trans_A,
                            const CBLAS_TRANSPOSE trans_B, const int_tp M,
                            const int_tp N, const int_tp K, const double alpha,
                            const double* A, const double* B, const double beta,
                            double* C) {
  int_tp lda = (trans_A == CblasNoTrans) ? K : M;
  int_tp ldb = (trans_B == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

}  // namespace caffe

