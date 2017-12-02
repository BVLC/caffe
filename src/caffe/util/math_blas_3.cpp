#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<half_fp>(const CBLAS_TRANSPOSE trans_a,
                          const CBLAS_TRANSPOSE trans_b, const int_tp M,
                          const int_tp N, const int_tp K,
                          const half_fp alpha,
                          const half_fp* a, const half_fp* b,
                          const half_fp beta,
                          half_fp* c) {
  int_tp inc_a = (trans_a == CblasNoTrans) ? 1 : M;
  int_tp inc_b = (trans_b == CblasNoTrans) ? N : 1;
  for (int_tp m = 0; m < M; m++) {
    for (int_tp n = 0; n < N; n++) {
      half_fp acc = 0;
      int_tp b_index = trans_b == CblasNoTrans ?
                       N : K * N;
      int_tp a_index = trans_a == CblasNoTrans ?
                       K * M : M;
      for (int_tp k = 0; k < K; k++) {
        acc += a[a_index] * b[b_index];
        a_index += inc_a;
        b_index += inc_b;
      }
      if (beta != 0)
        c[m * n + n] = acc * alpha + beta * c[m * n + n];
      else
        c[m * n + n] = acc * alpha;
    }
  }
}

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE trans_a,
                           const CBLAS_TRANSPOSE trans_b, const int_tp m,
                           const int_tp n, const int_tp k, const float alpha,
                           const float* a, const float* b, const float beta,
                           float* c) {
  int_tp lda = (trans_a == CblasNoTrans) ? k : m;
  int_tp ldb = (trans_b == CblasNoTrans) ? n : k;
  cblas_sgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb,
              beta, c, n);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b, const int_tp m,
                            const int_tp n, const int_tp k, const double alpha,
                            const double* a, const double* b, const double beta,
                            double* c) {
  int_tp lda = (trans_a == CblasNoTrans) ? k : m;
  int_tp ldb = (trans_b == CblasNoTrans) ? n : k;
  cblas_dgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb,
              beta, c, n);
}

}  // namespace caffe

