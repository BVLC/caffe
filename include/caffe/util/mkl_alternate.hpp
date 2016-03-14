#ifndef CAFFE_UTIL_MKL_ALTERNATE_H_
#define CAFFE_UTIL_MKL_ALTERNATE_H_

#ifdef USE_MKL

#include <mkl.h>

#else  // If use MKL, simply include the MKL header

#ifdef USE_EIGEN

#include <Eigen/Dense>

using Eigen::Map;
using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::InnerStride;

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

#define MAP_SVECTOR(name, ptr, N) Map<VectorXf> name(ptr, N)
#define MAP_CONST_SVECTOR(name, ptr, N) Map<const VectorXf> name(ptr, N)
#define MAP_CONST_SVECTOR_STRIDE(name, ptr, N, S) Map<const VectorXf, 0, InnerStride<> > name(ptr, N, InnerStride<>(S))
#define MAP_DVECTOR(name, ptr, N) Map<VectorXd> name(ptr, N)
#define MAP_CONST_DVECTOR(name, ptr, N) Map<const VectorXd> name(ptr, N)
#define MAP_CONST_DVECTOR_STRIDE(name, ptr, N, S) Map<const VectorXd, 0, InnerStride<> > name(ptr, N, InnerStride<>(S))
typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatXf;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatXd;

#define MAP_SMATRIX(name, ptr, M, N) Map<MatXf> name(ptr, M, N)
#define MAP_CONST_SMATRIX(name, ptr, M, N) Map<const MatXf> name(ptr, M, N)
#define MAP_DMATRIX(name, ptr, M, N) Map<MatXd> name(ptr, M, N)
#define MAP_CONST_DMATRIX(name, ptr, M, N) Map<const MatXd> name(ptr, M, N)

#else

extern "C" {
#include <cblas.h>
}

#endif  // USE_EIGEN

#include <math.h>

// Functions that caffe uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])
#define DEFINE_VSL_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, float* y) { \
    v##name<float>(n, a, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, double* y) { \
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
  void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, const float b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b));

// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float* b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, const double* b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i]);

#ifndef USE_EIGEN

// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
inline void cblas_saxpby(const int N, const float alpha, const float* X,
                         const int incX, const float beta, float* Y,
                         const int incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int N, const double alpha, const double* X,
                         const int incX, const double beta, double* Y,
                         const int incY) {
  cblas_dscal(N, beta, Y, incY);
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

#endif  // USE_EIGEN

#endif  // USE_MKL
#endif  // CAFFE_UTIL_MKL_ALTERNATE_H_
