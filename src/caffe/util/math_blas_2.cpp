#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemv<half_fp>(const CBLAS_TRANSPOSE trans_a,
                   const int_tp M, const int_tp N, const half_fp alpha,
                   const half_fp* a, const half_fp* X,
                   const half_fp beta, half_fp* Y) {
  int_tp a_inc = (trans_a == CblasNoTrans) ? 1 : N;
  int_tp y_cnt = (trans_a == CblasNoTrans) ? M : N;
  int_tp x_cnt = (trans_a == CblasNoTrans) ? N : M;
  for (int_tp m = 0; m < y_cnt; m++) {
    int_tp a_index = (trans_a == CblasNoTrans) ? M * N : M;
    half_fp acc = 0;
    for (int_tp n = 0; n < x_cnt; n++) {
      acc += a[a_index] * X[n];
      a_index += a_inc;
    }
    if (beta == 0)
      Y[m] = acc * alpha;
    else
      Y[m] = acc * alpha + beta * Y[m];
  }
}

template<>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE trans_a, const int_tp m,
                           const int_tp n, const float alpha, const float* a,
                           const float* X, const float beta, float* Y) {
  cblas_sgemv(CblasRowMajor, trans_a, m, n, alpha, a, n, X, 1, beta, Y, 1);
}

template<>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE trans_a, const int_tp m,
                            const int_tp n, const double alpha, const double* a,
                            const double* X, const double beta, double* Y) {
  cblas_dgemv(CblasRowMajor, trans_a, m, n, alpha, a, n, X, 1, beta, Y, 1);
}

}  // namespace caffe

