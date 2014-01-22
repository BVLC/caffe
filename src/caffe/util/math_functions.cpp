// Copyright 2013 Yangqing Jia

#include <limits>
//#include <mkl.h>
#include <eigen3/Eigen/Dense>
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <cublas_v2.h>
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Operations on aligned memory are faster than on unaligned memory.
// But unfortunately, the pointers passed in are not always aligned.
// Therefore, the memory-aligned Eigen::Map objects that wrap them
// cannot be assigned to. This happens in lrn_layer and makes
// test_lrn_layer crash with segmentation fault.
// TODO: Use aligned Eigen::Map when the pointer to be wrapped is aligned.

// Though the default map option is unaligned, making it explicit is no harm.
//const int data_alignment = Eigen::Aligned; // how is data allocated ?
const int data_alignment = Eigen::Unaligned;
typedef Eigen::Array<float, 1, Eigen::Dynamic> float_array_t;
typedef Eigen::Map<const float_array_t, data_alignment> const_map_vector_float_t;
typedef Eigen::Map<float_array_t, data_alignment> map_vector_float_t;
typedef Eigen::Array<double, 1, Eigen::Dynamic> double_array_t;
typedef Eigen::Map<const double_array_t, data_alignment> const_map_vector_double_t;
typedef Eigen::Map<double_array_t, data_alignment> map_vector_double_t;

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }


template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_copy<float>(const int N, const float* X, float* Y) {
  cblas_scopy(N, X, 1, Y, 1);
}

template <>
void caffe_copy<double>(const int N, const double* X, double* Y) {
  cblas_dcopy(N, X, 1, Y, 1);
}

template <>
void caffe_gpu_copy<float>(const int N, const float* X, float* Y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), N, X, 1, Y, 1));
}

template <>
void caffe_gpu_copy<double>(const int N, const double* X, double* Y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), N, X, 1, Y, 1));
}

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  // y := a*x + b*y
  //cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
  CHECK_GE(N, 0);
  CHECK(X);
  CHECK(Y);
  map_vector_float_t y_map(Y, N);
  // Eigen produces optimized code using lasy evaluation
  // http://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html
  y_map = const_map_vector_float_t(X, N) * alpha + y_map * beta;
}

template <>
void caffe_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
    // y := a*x + b*y
  //cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
  CHECK_GE(N, 0);
  CHECK(X);
  CHECK(Y);
  map_vector_double_t y_map(Y, N);
  y_map = const_map_vector_double_t(X, N) * alpha + y_map * beta;
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  //vsAdd(n, a, b, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(b);
  CHECK(y);
  map_vector_float_t(y, n) = const_map_vector_float_t(a, n) +
      const_map_vector_float_t(b, n);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  //vdAdd(n, a, b, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(b);
  CHECK(y);
  map_vector_double_t(y, n) = const_map_vector_double_t(a, n) +
      const_map_vector_double_t(b, n);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  //vsSub(n, a, b, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(b);
  CHECK(y);
  map_vector_float_t(y, n) = const_map_vector_float_t(a, n) -
      const_map_vector_float_t(b, n);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  //vdSub(n, a, b, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(b);
  CHECK(y);
  map_vector_double_t(y, n) = const_map_vector_double_t(a, n) -
      const_map_vector_double_t(b, n);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  //vsMul(n, a, b, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(b);
  CHECK(y);
  map_vector_float_t(y, n) = const_map_vector_float_t(a, n) *
       const_map_vector_float_t(b, n);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  //vdMul(n, a, b, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(b);
  CHECK(y);
  map_vector_double_t(y, n) = const_map_vector_double_t(a, n) *
      const_map_vector_double_t(b, n);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  //vsDiv(n, a, b, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(b);
  CHECK(y);
  map_vector_float_t(y, n) = const_map_vector_float_t(a, n) /
      const_map_vector_float_t(b, n);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  //vdDiv(n, a, b, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(b);
  CHECK(y);
  map_vector_double_t(y, n) = const_map_vector_double_t(a, n) /
      const_map_vector_double_t(b, n);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  //vsPowx(n, a, b, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(y);
  map_vector_float_t(y, n) = const_map_vector_float_t(a, n).pow(b);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  //vdPowx(n, a, b, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(y);
  map_vector_double_t(y, n) = const_map_vector_double_t(a, n).pow(b);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  // http://software.intel.com/sites/products/documentation/hpc/mkl/mklman/GUID-F003F826-81BF-42EC-AE51-2EF624893133.htm
  // v?Sqr Performs element by element squaring of the vector.
  //vsSqr(n, a, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(y);
  caffe_powx<float>(n, a, 2, y);
  // TODO: which is faster?
//  map_vector_float_t(y, n) = const_map_vector_float_t(a, n) *
//      const_map_vector_float_t(a, n);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  //vdSqr(n, a, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(y);
  caffe_powx<double>(n, a, 2, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  //vsExp(n, a, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(y);
  map_vector_float_t(y, n) = const_map_vector_float_t(a, n).exp();
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  //vdExp(n, a, y);
  CHECK_GE(n, 0);
  CHECK(a);
  CHECK(y);
  map_vector_double_t(y, n) = const_map_vector_double_t(a, n).exp();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype, Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_vRngUniform(const int n, Dtype* r,
    const Dtype a, const Dtype b) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  //VSL_CHECK(vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, Caffe::vsl_stream(),
  //    n, r, a, b));

  // FIXME check if boundaries are handled in the same way ?
  // Fixed by caffe_nextafter
  boost::random::uniform_real_distribution<Dtype> random_distribution(
      a, caffe_nextafter<Dtype>(b));
  Caffe::random_generator_t &generator = Caffe::vsl_stream();

  for(int i = 0; i < n; i += 1) {
    r[i] = random_distribution(generator);
  }
}

template
void caffe_vRngUniform<float>(const int n, float* r,
                                       const float a, const float b);
template
void caffe_vRngUniform<double>(const int n, double* r,
                                       const double a, const double b);

template <typename Dtype>
void caffe_vRngGaussian(const int n, Dtype* r, const Dtype a,
    const Dtype sigma) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  //VSL_CHECK(vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
//      Caffe::vsl_stream(), n, r, a, sigma));

    // FIXME check if parameters are handled in the same way ?
    // http://www.boost.org/doc/libs/1_55_0/doc/html/boost/random/normal_distribution.html
    // http://software.intel.com/sites/products/documentation/hpc/mkl/mklman/GUID-63196F25-5013-4038-8BCD-2613C4EF3DE4.htm
    // The above two documents show that the probability density functions are different.
    // But the unit tests still pass. Maybe their codes are the same or
    // the tests are irrelevant to the random numbers.
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  Caffe::random_generator_t &generator = Caffe::vsl_stream();

  for(int i = 0; i < n; i += 1) {
    r[i] = random_distribution(generator);
  }
}

template
void caffe_vRngGaussian<float>(const int n, float* r, const float a,
    const float sigma);

template
void caffe_vRngGaussian<double>(const int n, double* r, const double a,
    const double sigma);

template <typename Dtype>
void caffe_vRngBernoulli(const int n, Dtype* r, const double p) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
    // FIXME check if parameters are handled in the same way ?
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  Caffe::random_generator_t &generator = Caffe::vsl_stream();

  for(int i = 0; i < n; i += 1) {
    r[i] = random_distribution(generator);
  }
}

template
void caffe_vRngBernoulli<int>(const int n, int* r, const double p);

template <>
float caffe_cpu_dot<float>(const int n, const float* x, const float* y) {
  return cblas_sdot(n, x, 1, y, 1);
}

template <>
double caffe_cpu_dot<double>(const int n, const double* x, const double* y) {
  return cblas_ddot(n, x, 1, y, 1);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

}  // namespace caffe
