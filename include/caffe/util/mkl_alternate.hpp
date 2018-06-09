#ifndef CAFFE_UTIL_MKL_ALTERNATE_H_
#define CAFFE_UTIL_MKL_ALTERNATE_H_

#ifdef USE_MKL

#include <mkl.h>

#else  // If use MKL, simply include the MKL header

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif  // USE_ACCELERATE

#include <math.h>
#include "caffe/util/half_fp.hpp"

// Functions that caffe uses but are not present if MKL is not linked.

// a simple way to define the vsl unary functions. The operation should
// be in the form e.g. Y[i] = sqrt(a[i])
#define DEFINE_VSL_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int_tp n, const Dtype* a, Dtype* Y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(Y); \
    for (int_tp i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int_tp n, const float* a, float* Y) { \
    v##name<float>(n, a, Y); \
  } \
  inline void vd##name( \
      const int_tp n, const double* a, double* Y) { \
    v##name<double>(n, a, Y); \
  }

DEFINE_VSL_UNARY_FUNC(Sqr, Y[i] = a[i] * a[i])
DEFINE_VSL_UNARY_FUNC(Sqrt, Y[i] = std::sqrt(a[i]))
DEFINE_VSL_UNARY_FUNC(Exp, Y[i] = std::exp(a[i]))
DEFINE_VSL_UNARY_FUNC(Ln, Y[i] = std::log(a[i]))
DEFINE_VSL_UNARY_FUNC(Abs, Y[i] = std::fabs(a[i]))

// a simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. Y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
  template<typename Dtype> \
  void v##name(const int_tp n, const Dtype* a, const Dtype b, Dtype* Y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(Y); \
    for (int_tp i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int_tp n, const float* a, const float b, float* Y) { \
    v##name<float>(n, a, b, Y); \
  } \
  inline void vd##name( \
      const int_tp n, const double* a, const float b, double* Y) { \
    v##name<double>(n, a, b, Y); \
  }

DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, Y[i] = pow(a[i], b))

// a simple way to define the vsl binary functions. The operation should
// be in the form e.g. Y[i] = a[i] + b[i]
#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int_tp n, const Dtype* a, const Dtype* b, Dtype* Y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(Y); \
    for (int_tp i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int_tp n, const float* a, const float* b, float* Y) { \
    v##name<float>(n, a, b, Y); \
  } \
  inline void vd##name( \
      const int_tp n, const double* a, const double* b, double* Y) { \
    v##name<double>(n, a, b, Y); \
  }

DEFINE_VSL_BINARY_FUNC(Add, Y[i] = a[i] + b[i])
DEFINE_VSL_BINARY_FUNC(Sub, Y[i] = a[i] - b[i])
DEFINE_VSL_BINARY_FUNC(Mul, Y[i] = a[i] * b[i])
DEFINE_VSL_BINARY_FUNC(Div, Y[i] = a[i] / b[i])

// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
inline void cblas_haxpby(const int_tp n, const caffe::half_fp alpha,
                         const caffe::half_fp* X,
                         const int_tp incX, const caffe::half_fp beta,
                         caffe::half_fp* Y,
                         const int_tp incY) {
  for (int_tp i = 0; i < n; ++i)
    Y[i * incY] *= beta;

  for (int_tp i = 0; i < n; ++i) {
    Y[i * incY] += alpha * X[i * incX];
  }
}
inline void cblas_saxpby(const int_tp n, const float alpha, const float* X,
                         const int_tp incX, const float beta, float* Y,
                         const int_tp incY) {
  cblas_sscal(n, beta, Y, incY);
  cblas_saxpy(n, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int_tp n, const double alpha, const double* X,
                         const int_tp incX, const double beta, double* Y,
                         const int_tp incY) {
  cblas_dscal(n, beta, Y, incY);
  cblas_daxpy(n, alpha, X, incX, Y, incY);
}

#endif  // USE_MKL
#endif  // CAFFE_UTIL_MKL_ALTERNATE_H_
