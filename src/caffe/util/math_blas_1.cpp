#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

// AXPY
template<>
void caffe_axpy<float>(const int_tp n, const float alpha, const float* X,
                       float* Y) {
  cblas_saxpy(n, alpha, X, 1, Y, 1);
}
template<>
void caffe_axpy<double>(const int_tp n, const double alpha, const double* X,
                        double* Y) {
  cblas_daxpy(n, alpha, X, 1, Y, 1);
}
template<typename Dtype>
void caffe_axpy(const int_tp n, const Dtype alpha,
                         const Dtype* X, Dtype* Y) {
#pragma omp parallel for
  for (int_tp i = 0; i < n; ++i) {
    Y[i] += alpha * X[i];
  }
}
template void caffe_axpy<half_fp>(const int_tp n, const half_fp alpha,
                         const half_fp* X, half_fp* Y);
template void caffe_axpy<int8_t>(const int_tp n, const int8_t alpha,
                         const int8_t* X, int8_t* Y);
template void caffe_axpy<int16_t>(const int_tp n, const int16_t alpha,
                         const int16_t* X, int16_t* Y);
template void caffe_axpy<int32_t>(const int_tp n, const int32_t alpha,
                         const int32_t* X, int32_t* Y);
template void caffe_axpy<int64_t>(const int_tp n, const int64_t alpha,
                         const int64_t* X, int64_t* Y);

// AXPBY
template<>
void caffe_axpby<half_fp>(const int_tp n,
                        const half_fp alpha, const half_fp* X,
                        const half_fp beta, half_fp* Y) {
  cblas_haxpby(n, alpha, X, 1, beta, Y, 1);
}
template<>
void caffe_axpby<float>(const int_tp n, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(n, alpha, X, 1, beta, Y, 1);
}
template<>
void caffe_axpby<double>(const int_tp n, const double alpha,
                             const double* X, const double beta, double* Y) {
  cblas_daxpby(n, alpha, X, 1, beta, Y, 1);
}
template<typename Dtype>
void caffe_axpby(const int_tp n, const Dtype alpha, const Dtype* X,
                 const Dtype beta, Dtype* Y) {
#pragma omp parallel for
  for (int_tp i = 0; i < n; ++i) {
    Y[i] = alpha * X[i] + beta * Y[i];
  }
}
template void caffe_axpby<half_fp>(const int_tp n, const half_fp alpha,
                         const half_fp* X, const half_fp beta, half_fp* Y);
template void caffe_axpby<int8_t>(const int_tp n, const int8_t alpha,
                         const int8_t* X, const int8_t beta, int8_t* Y);
template void caffe_axpby<int16_t>(const int_tp n, const int16_t alpha,
                         const int16_t* X, const int16_t beta, int16_t* Y);
template void caffe_axpby<int32_t>(const int_tp n, const int32_t alpha,
                         const int32_t* X, const int32_t beta, int32_t* Y);
template void caffe_axpby<int64_t>(const int_tp n, const int64_t alpha,
                         const int64_t* X, const int64_t beta, int64_t* Y);

// DOT
template<typename Dtype>
Dtype caffe_strided_dot(const int_tp n,
                                const Dtype* x, const int_tp incx,
                                const Dtype* y, const int_tp incy) {
  half_fp sum = 0;
  for (int_tp i = 0; i < n; i++) {
    sum += x[i * incx] * y[i * incy];
  }
  return sum;
}

template half_fp caffe_strided_dot(const int_tp n,
                                   const half_fp* x, const int_tp incx,
                                   const half_fp* y, const int_tp incy);
template int8_t caffe_strided_dot(const int_tp n,
                                  const int8_t* x, const int_tp incx,
                                  const int8_t* y, const int_tp incy);
template int16_t caffe_strided_dot(const int_tp n,
                                   const int16_t* x, const int_tp incx,
                                   const int16_t* y, const int_tp incy);
template int32_t caffe_strided_dot(const int_tp n,
                                   const int32_t* x, const int_tp incx,
                                   const int32_t* y, const int_tp incy);
template int64_t caffe_strided_dot(const int_tp n,
                                   const int64_t* x, const int_tp incx,
                                   const int64_t* y, const int_tp incy);

template<>
float caffe_strided_dot<float>(const int_tp n, const float* X,
                               const int_tp incx, const float* Y,
                               const int_tp incy) {
  return cblas_sdot(n, X, incx, Y, incy);
}

template<>
double caffe_strided_dot<double>(const int_tp n, const double* X,
                                 const int_tp incx, const double* Y,
                                 const int_tp incy) {
  return cblas_ddot(n, X, incx, Y, incy);
}

template<typename Dtype>
Dtype caffe_dot(const int_tp n, const Dtype* X, const Dtype* Y) {
  Dtype r;
  #pragma omp for reduction(+ : r)
  for (size_t i = 0; i < n; i++) {
    r += X[i] * Y[i];
  }
  return r;
}

template half_fp caffe_dot<half_fp>(const int_tp n,
                               const half_fp* X, const half_fp* Y);
template int8_t caffe_dot<int8_t>(const int_tp n, const int8_t* X,
                              const int8_t* Y);
template int16_t caffe_dot<int16_t>(const int_tp n, const int16_t* X,
                              const int16_t* Y);
template int32_t caffe_dot<int32_t>(const int_tp n, const int32_t* X,
                              const int32_t* Y);
template int64_t caffe_dot<int64_t>(const int_tp n, const int64_t* X,
                              const int64_t* Y);

template<>
float caffe_dot(const int_tp n, const float* X, const float* Y) {
  return caffe_strided_dot(n, X, 1, Y, 1);
}
template<>
double caffe_dot(const int_tp n, const double* X, const double* Y) {
  return caffe_strided_dot(n, X, 1, Y, 1);
}

// ASUM
template<>
half_fp caffe_asum<half_fp>(const int_tp n, const half_fp* x) {
  half_fp sum = 0;
  for (int_tp i = 0; i < n; i++)
    sum += fabs(x[i]);
  return sum;
}
template<>
float caffe_asum<float>(const int_tp n, const float* x) {
  return cblas_sasum(n, x, 1);
}
template<>
double caffe_asum<double>(const int_tp n, const double* x) {
  return cblas_dasum(n, x, 1);
}
template<typename Dtype>
Dtype caffe_asum(const int_tp n, const Dtype* x) {
  Dtype sum = 0;
  for (int_tp i = 0; i < n; i++)
    sum += abs(x[i]);
  return sum;
}
template int8_t caffe_asum<int8_t>(const int_tp n, const int8_t* x);
template int16_t caffe_asum<int16_t>(const int_tp n, const int16_t* x);
template int32_t caffe_asum<int32_t>(const int_tp n, const int32_t* x);
template int64_t caffe_asum<int64_t>(const int_tp n, const int64_t* x);

// SCALE
template<typename Dtype>
void caffe_scale(const int_tp n, const Dtype alpha, const Dtype *X,
                 Dtype* Y) {
  for (int_tp i = 0; i < n; i++) {
    Y[i] = X[i];
  }
  caffe_scal(n, alpha, Y);
}

template void caffe_scale<half_fp>(const int_tp n, const half_fp alpha,
                                   const half_fp *X, half_fp* Y);
template void caffe_scale<int8_t>(const int_tp n, const int8_t alpha,
                                   const int8_t *X, int8_t* Y);
template void caffe_scale<int16_t>(const int_tp n, const int16_t alpha,
                                   const int16_t *X, int16_t* Y);
template void caffe_scale<int32_t>(const int_tp n, const int32_t alpha,
                                   const int32_t *X, int32_t* Y);
template void caffe_scale<int64_t>(const int_tp n, const int64_t alpha,
                                   const int64_t *X, int64_t* Y);

template<>
void caffe_scale<float>(const int_tp n, const float alpha, const float *X,
                            float* Y) {
  cblas_scopy(n, X, 1, Y, 1);
  cblas_sscal(n, alpha, Y, 1);
}

template<>
void caffe_scale<double>(const int_tp n, const double alpha,
                             const double *X, double* Y) {
  cblas_dcopy(n, X, 1, Y, 1);
  cblas_dscal(n, alpha, Y, 1);
}

// SCAL
template<>
void caffe_scal<float>(const int_tp n, const float alpha, float *X) {
  cblas_sscal(n, alpha, X, 1);
}
template<>
void caffe_scal<double>(const int_tp n, const double alpha, double *X) {
  cblas_dscal(n, alpha, X, 1);
}
template<typename Dtype>
void caffe_scal(const int_tp n, const Dtype alpha, Dtype *X) {
#pragma omp parallel for
  for (int_tp i = 0; i < n; ++i) {
    X[i] *= alpha;
  }
}
template void caffe_scal<half_fp>(const int_tp n, const half_fp alpha,
                                  half_fp *X);
template void caffe_scal<int8_t>(const int_tp n, const int8_t alpha,
                                 int8_t *X);
template void caffe_scal<int16_t>(const int_tp n, const int16_t alpha,
                                  int16_t *X);
template void caffe_scal<int32_t>(const int_tp n, const int32_t alpha,
                                  int32_t *X);
template void caffe_scal<int64_t>(const int_tp n, const int64_t alpha,
                                  int64_t *X);

}  // namespace caffe

