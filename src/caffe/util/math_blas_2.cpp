#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_gemv<half_fp>(const CBLAS_TRANSPOSE trans_A,
                   const int_tp M, const int_tp N, const half_fp alpha,
                   const half_fp* a, const half_fp* x,
                   const half_fp beta, half_fp* y) {
  int_tp a_inc = (trans_A == CblasNoTrans) ? 1 : N;
  int_tp y_cnt = (trans_A == CblasNoTrans) ? M : N;
  int_tp x_cnt = (trans_A == CblasNoTrans) ? N : M;
  for (int_tp m = 0; m < y_cnt; m++) {
    int_tp a_index = (trans_A == CblasNoTrans) ? m * N : m;
    half_fp acc = 0;
    for (int_tp n = 0; n < x_cnt; n++) {
      acc += a[a_index] * x[n];
      a_index += a_inc;
    }
    if (beta == 0)
      y[m] = acc * alpha;
    else
      y[m] = acc * alpha + beta * y[m];
  }
}

template<>
void caffe_gemv<float>(const CBLAS_TRANSPOSE trans_A, const int_tp M,
                           const int_tp N, const float alpha, const float* A,
                           const float* x, const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, trans_A, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template<>
void caffe_gemv<double>(const CBLAS_TRANSPOSE trans_A, const int_tp M,
                            const int_tp N, const double alpha, const double* A,
                            const double* x, const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, trans_A, M, N, alpha, A, N, x, 1, beta, y, 1);
}

}  // namespace caffe

