#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/arraymath.hpp"

namespace caffe {
namespace arraymath_detail {

/****** Unary array operations ******/
template <typename T, typename F>
__global__ void evalUnary_kernel(const int n, const T *x, T *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = F::eval(x[index]);
  }
}
template<typename T, typename F>
void evalUnary_gpu(int N, const T *a, T *r) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  evalUnary_kernel<T, F> <<< CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, r);
  CUDA_POST_KERNEL_CHECK;
}

#define DEFINE_UNARY_OP(name) template<typename T> struct unary_##name {\
  __device__ static inline T eval( const T& v ) { return name(v); }\
};

#define IMPLEMENT_UNARY(name) DEFINE_UNARY_OP(name)\
template<typename T> void name##_gpu(int N, const T* a, T* r) {\
  evalUnary_gpu<T, arraymath_detail::unary_##name<T> >(N, a, r);\
}\
template void name##_gpu<float>(int N, const float* a, float* r);\
template void name##_gpu<double>(int N, const double* a, double* r);

// Implement common unary functions
template<typename T>
__device__ T negate(const T &v) { return -v; }
template<typename T>
__device__ T sign(const T &v) { return v < 0 ? -1 : (v > 0 ? 1 : 0); }

IMPLEMENT_UNARY(abs);
IMPLEMENT_UNARY(exp);
IMPLEMENT_UNARY(log);
IMPLEMENT_UNARY(negate);
IMPLEMENT_UNARY(sign);
IMPLEMENT_UNARY(sqrt);

/****** Binary array operations ******/
template <typename T, typename F>
__global__ void evalBinary_kernel(const int n, const T *a, const T *b, T *r) {
  CUDA_KERNEL_LOOP(index, n) {
    r[index] = F::eval(a[index], b[index]);
  }
}
template <typename T, typename F>
__global__ void evalBinary_kernel(const int n, T a, const T *b, T *r) {
  CUDA_KERNEL_LOOP(index, n) {
    r[index] = F::eval(a, b[index]);
  }
}
template <typename T, typename F>
__global__ void evalBinary_kernel(const int n, const T *a, T b, T *r) {
  CUDA_KERNEL_LOOP(index, n) {
    r[index] = F::eval(a[index], b);
  }
}
template<typename T, typename F, typename T1, typename T2>
void evalBinary_gpu(int N, T1 a, T2 b, T *r) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  evalBinary_kernel<T, F> <<< CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, r);
  CUDA_POST_KERNEL_CHECK;
}

#define DEFINE_BINARY_OP(name, OP) template<typename T> struct binary_##name {\
  __device__ static inline T eval(T a, T b) { return a OP b; }\
};
#define DEFINE_BINARY_F(name, F) template<typename T> struct binary_##name {\
  __device__ static inline T eval(T a, T b) { return F(a, b); }\
};

#define IMPLEMENT_BINARY_T(name, T1, T2) \
template<typename T> void name##_gpu(int N, const T T1 a, const T T2 b, T* r) {\
  evalBinary_gpu<T, binary_##name<T> >(N, a, b, r);\
}\
template void name##_gpu<float>(int N, const float T1 a, const float T2 b, \
  float* r);\
template void name##_gpu<double>(int N, const double T1 a, const double T2 b, \
  double* r);

#define IMPLEMENT_BINARY(name) IMPLEMENT_BINARY_T(name, *, *)\
IMPLEMENT_BINARY_T(name, &, *)\
IMPLEMENT_BINARY_T(name, *, &)

#define IMPLEMENT_BINARY_OP(name, OP) DEFINE_BINARY_OP(name, OP)\
IMPLEMENT_BINARY(name)

#define IMPLEMENT_BINARY2(name, F) DEFINE_BINARY_F(name, F)\
IMPLEMENT_BINARY(name)

// Implement common unary functions
IMPLEMENT_BINARY_OP(add, +);
IMPLEMENT_BINARY_OP(sub, -);
IMPLEMENT_BINARY_OP(mul, *);
IMPLEMENT_BINARY_OP(div, /);
IMPLEMENT_BINARY2(maximum, max);
IMPLEMENT_BINARY2(minimum, min);
IMPLEMENT_BINARY2(pow, pow);



/**** gemm ****/
void gemm_gpu(bool tA, bool tB, const int M, const int N, const int K,
  float alpha, const float *A, int lda, const float *B, int ldb,
  float beta, float *C, int ldc) {
  // Note that cublas follows fortran order.
  cublasOperation_t cuTransA = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = tB ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}
void gemm_gpu(bool tA, bool tB, const int M, const int N, const int K,
  double alpha, const double *A, int lda, const double *B, int ldb,
  double beta, double *C, int ldc) {
  // Note that cublas follows fortran order.
  cublasOperation_t cuTransA = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = tB ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}
}  // namespace arraymath_detail
}  // namespace caffe
