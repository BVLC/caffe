#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<half_fp>(const CBLAS_TRANSPOSE trans_A,
                          const CBLAS_TRANSPOSE trans_B, const int_tp M,
                          const int_tp N, const int_tp K,
                          const half_fp alpha,
                          const half_fp* A, const half_fp* B,
                          const half_fp beta,
                          half_fp* C) {
  int_tp inc_a = (trans_A == CblasNoTrans) ? 1 : M;
  int_tp inc_b = (trans_B == CblasNoTrans) ? N : 1;
  for (int_tp m = 0; m < M; ++m) {
    for (int_tp n = 0; n < N; ++n) {
      half_fp acc = 0;
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

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE trans_A,
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
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE trans_A,
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

