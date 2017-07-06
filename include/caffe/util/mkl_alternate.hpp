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

// Functions that caffe uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])
#define DEFINE_VSL_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int_tp n, const Dtype* a, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int_tp i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int_tp n, const float* a, float* y) { \
    v##name<float>(n, a, y); \
  } \
  inline void vd##name( \
      const int_tp n, const double* a, double* y) { \
    v##name<double>(n, a, y); \
  }

DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i]);
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]));
DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]));
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]));

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
  template<typename Dtype> \
  void v##name(const int_tp n, const Dtype* a, const Dtype b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int_tp i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int_tp n, const float* a, const float b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int_tp n, const double* a, const float b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b));

// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int_tp n, const Dtype* a, const Dtype* b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
    for (int_tp i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int_tp n, const float* a, const float* b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int_tp n, const double* a, const double* b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i]);

// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
#ifdef HAS_HALF_SUPPORT
inline void cblas_haxpby(const int_tp N, const half alpha, const half* X,
                         const int_tp incX, const half beta, half* Y,
                         const int_tp incY) {
  for (int_tp n = 0; n < N; n++)
    Y[n * incY] *= beta;

  for (int_tp n = 0; n < N; n++) {
    Y[n * incY] += alpha * X[n * incX];
  }
}
#endif

inline void cblas_saxpby(const int_tp N, const float alpha, const float* X,
                         const int_tp incX, const float beta, float* Y,
                         const int_tp incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int_tp N, const double alpha, const double* X,
                         const int_tp incX, const double beta, double* Y,
                         const int_tp incY) {
  cblas_dscal(N, beta, Y, incY);
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

#endif  // USE_MKL
#endif  // CAFFE_UTIL_MKL_ALTERNATE_H_
