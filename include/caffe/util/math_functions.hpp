// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

// Decaf gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

unsigned int caffe_rng_rand();

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                        Dtype* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r);

template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
int caffe_cpu_hamming_distance(const int n, const Dtype* x, const Dtype* y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype caffe_cpu_asum(const int n, const Dtype* x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template<typename Dtype>
inline char caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(x); CHECK(y); \
    for (int i = 0; i < n; ++i) { \
      operation; \
    } \
  }

#define INSTANTIATE_CAFFE_CPU_UNARY_FUNC(name) \
  template <> \
  void caffe_cpu_##name<float>(const int n, const float* x, float* y); \
  template <> \
  void caffe_cpu_##name<double>(const int n, const double* x, double* y)

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]));

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro
bool caffe_signbit(float arg);
bool caffe_signbit(double arg);
bool caffe_signbit(long double arg);
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, y[i] = caffe_signbit(x[i]));

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]));

template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

#ifndef CPU_ONLY  // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void caffe_gpu_rng_uniform(const int n, unsigned int* r);

// caffe_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                            Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
uint32_t caffe_gpu_hamming_distance(const int n, const Dtype* x,
                                    const Dtype* y);

template <typename Dtype>
void caffe_gpu_asum(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sign(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_fabs(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

#endif  // CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
