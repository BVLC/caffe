/*
 * greentea_math_functions.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/greentea/greentea_math_functions.hpp"
#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"

// TODO: Remove:

#ifdef USE_CLBLAS
#include <clBLAS.h>
#endif USE_CLBLAS

namespace caffe {

// Copy from OpenCL buffer to main memory
void greentea_gpu_memcpy(const size_t N, const cl_mem X, void *Y,
                         viennacl::ocl::context &ctx) {
  if (Y != NULL) {
    cl_int err = clEnqueueReadBuffer(ctx.get_queue().handle().get(), X, CL_TRUE,
                                     0, N, Y, 0, NULL, NULL);
  }
  ctx.get_queue().finish();
}

// Copy from main memory to OpenCL buffer
void greentea_gpu_memcpy(const size_t N, const void* X, cl_mem Y,
                         viennacl::ocl::context &ctx) {
  if (X != NULL) {
    cl_int err = clEnqueueWriteBuffer(ctx.get_queue().handle().get(), Y,
    CL_TRUE,
                                      0, N, X, 0, NULL, NULL);
  }
  ctx.get_queue().finish();
}

// Copy from OpenCL buffer to OpenCL buffer
template<typename Dtype>
void greentea_copy(const int N, const cl_mem X, cl_mem Y,
                   viennacl::ocl::context &ctx) {
  if (X != Y) {
    cl_int err = clEnqueueCopyBuffer(ctx.get_queue().handle().get(), X, Y, 0, 0,
                                     sizeof(Dtype) * N, 0, NULL, NULL);
  }
  ctx.get_queue().finish();
}

// Explicit instantiations
template void greentea_copy<int>(const int N, const cl_mem X, cl_mem Y,
                                 viennacl::ocl::context &ctx);
template void greentea_copy<unsigned int>(const int N, const cl_mem X, cl_mem Y,
                                          viennacl::ocl::context &ctx);
template void greentea_copy<float>(const int N, const cl_mem X, cl_mem Y,
                                   viennacl::ocl::context &ctx);
template void greentea_copy<double>(const int N, const cl_mem X, cl_mem Y,
                                    viennacl::ocl::context &ctx);

template<class Dtype>
void greentea_gpu_gemm(int ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int M, const int N,
                       const int K, const Dtype alpha, const cl_mem A, int offA,
                       const cl_mem B, int offB, const Dtype beta, cl_mem C,
                       int offC) {

  int offArow = offA;
  int offAcol = 0;
  int incArow = 1;
  int incAcol = 1;
  int offBrow = offB;
  int offBcol = 0;
  int incBrow = 1;
  int incBcol = 1;
  int offCrow = offC;
  int offCcol = 0;
  int incCrow = 1;
  int incCcol = 1;

  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;

#ifdef USE_VIENNACLBLAS
  ViennaCLBackend backend;
  ViennaCLBackendCreate(&backend);
  ViennaCLBackendSetOpenCLContextID(backend, static_cast<ViennaCLInt>(ctx_id));

  ViennaCLTranspose vclTransA =
      (TransA == CblasNoTrans) ? ViennaCLNoTrans : ViennaCLTrans;
  ViennaCLTranspose vclTransB =
      (TransB == CblasNoTrans) ? ViennaCLNoTrans : ViennaCLTrans;

  ViennaCLOrder vclOrderA = ViennaCLRowMajor;
  ViennaCLOrder vclOrderB = ViennaCLRowMajor;
  ViennaCLOrder vclOrderC = ViennaCLRowMajor;

  if (std::is_same<Dtype, float>::value) {
    GREENTEA_BLAS_CHECK(
        ViennaCLOpenCLSgemm(backend, vclOrderA, vclTransA, vclOrderB, vclTransB,
                            vclOrderC, M, N, K, alpha, A, offArow, offAcol,
                            incArow, incAcol, lda, B, offBrow, offBcol, incBrow,
                            incBcol, ldb, beta, C, offCrow, offCcol, incCrow,
                            incCcol, ldc));
  } else {
    GREENTEA_BLAS_CHECK(
        ViennaCLOpenCLDgemm(backend, vclOrderA, vclTransA, vclOrderB, vclTransB,
                            vclOrderC, M, N, K, alpha, A, offArow, offAcol,
                            incArow, incAcol, lda, B, offBrow, offBcol, incBrow,
                            incBcol, ldb, beta, C, offCrow, offCcol, incCrow,
                            incCcol, ldc));
  }
#endif
#ifdef USE_CLBLAS
  clblasOrder clOrder = clblasRowMajor;
  clblasTranspose clTransA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  clblasTranspose clTransB =
      (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;

  viennacl::ocl::context ctx = viennacl::ocl::get_context(ctx_id);

  cl_command_queue queue = ctx.get_queue().handle().get();

  if (std::is_same<Dtype, float>::value) {
    clblasSgemm(clOrder, clTransA, clTransB, M, N, K, alpha, A, offArow, lda, B,
                offBrow, ldb, beta, C, offCrow, ldc, 1, &queue, 0, NULL, NULL);
  } else {
    clblasDgemm(clOrder, clTransA, clTransB, M, N, K, alpha, A, offArow, lda, B,
                offBrow, ldb, beta, C, offCrow, ldc, 1, &queue, 0, NULL, NULL);
  }
#endif

}

template void greentea_gpu_gemm<float>(int ctx_id, const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int M, const int N, const int K,
                                       const float alpha, const cl_mem A,
                                       int offA, const cl_mem B, int offB,
                                       const float beta, cl_mem C, int offC);
template void greentea_gpu_gemm<double>(int ctx_id, const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int M, const int N, const int K,
                                       const double alpha, const cl_mem A,
                                       int offA, const cl_mem B, int offB,
                                       const double beta, cl_mem C, int offC);

/*  template<>
 void greentea_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
 const CBLAS_TRANSPOSE TransB, const int M,
 const int N, const int K, const double alpha,
 const double* A, const double* B,
 const double beta, double* C) {
 // Note that cublas follows fortran order.
 int lda = (TransA == CblasNoTrans) ? K : M;
 int ldb = (TransB == CblasNoTrans) ? N : K;
 cublasOperation_t cuTransA =
 (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
 cublasOperation_t cuTransB =
 (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
 CUBLAS_CHECK(
 cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA, N, M, K, &alpha,
 B, ldb, A, lda, &beta, C, N));
 }

 template<>
 void greentea_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
 const int N, const float alpha, const float* A,
 const float* x, const float beta, float* y) {
 cublasOperation_t cuTransA =
 (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
 CUBLAS_CHECK(
 cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha, A, N, x, 1,
 &beta, y, 1));
 }

 template<>
 void greentea_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
 const int N, const double alpha, const double* A,
 const double* x, const double beta, double* y) {
 cublasOperation_t cuTransA =
 (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
 CUBLAS_CHECK(
 cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha, A, N, x, 1,
 &beta, y, 1));
 }

 template<>
 void greentea_gpu_axpy<float>(const int N, const float alpha, const float* X,
 float* Y) {
 CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
 }

 template<>
 void greentea_gpu_axpy<double>(const int N, const double alpha, const double* X,
 double* Y) {
 CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
 }

 void greentea_gpu_memcpy(const size_t N, const void* X, void* Y) {
 if (X != Y) {
 CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
 }
 }

 template<>
 void greentea_gpu_scal<float>(const int N, const float alpha, float *X) {
 CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
 }

 template<>
 void greentea_gpu_scal<double>(const int N, const double alpha, double *X) {
 CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
 }

 template<>
 void greentea_gpu_axpby<float>(const int N, const float alpha, const float* X,
 const float beta, float* Y) {
 greentea_gpu_scal<float>(N, beta, Y);
 greentea_gpu_axpy<float>(N, alpha, X, Y);
 }

 template<>
 void greentea_gpu_axpby<double>(const int N, const double alpha,
 const double* X, const double beta, double* Y) {
 greentea_gpu_scal<double>(N, beta, Y);
 greentea_gpu_axpy<double>(N, alpha, X, Y);
 }

 template<>
 void greentea_gpu_dot<float>(const int n, const float* x, const float* y,
 float* out) {
 CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
 }

 template<>
 void greentea_gpu_dot<double>(const int n, const double* x, const double* y,
 double * out) {
 CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
 }

 template<>
 void greentea_gpu_asum<float>(const int n, const float* x, float* y) {
 CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
 }

 template<>
 void greentea_gpu_asum<double>(const int n, const double* x, double* y) {
 CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
 }

 template<>
 void greentea_gpu_scale<float>(const int n, const float alpha, const float *x,
 float* y) {
 CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
 CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
 }

 template<>
 void greentea_gpu_scale<double>(const int n, const double alpha,
 const double *x, double* y) {
 CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
 CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
 }

 template <typename Dtype>
 __global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
 CUDA_KERNEL_LOOP(index, n) {
 y[index] = alpha;
 }
 }

 template<typename Dtype>
 void greentea_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
 if (alpha == 0) {
 CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
 return;
 }
 // NOLINT_NEXT_LINE(whitespace/operators)
 set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, alpha, Y);
 }

 template void greentea_gpu_set<int>(const int N, const int alpha, int* Y);
 template void greentea_gpu_set<float>(const int N, const float alpha,
 float* Y);
 template void greentea_gpu_set<double>(const int N, const double alpha,
 double* Y);

 template <typename Dtype>
 __global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
 CUDA_KERNEL_LOOP(index, n) {
 y[index] += alpha;
 }
 }

 template<>
 void greentea_gpu_add_scalar(const int N, const float alpha, float* Y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, alpha, Y);
 }

 template<>
 void greentea_gpu_add_scalar(const int N, const double alpha, double* Y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, alpha, Y);
 }

 template <typename Dtype>
 __global__ void add_kernel(const int n, const Dtype* a,
 const Dtype* b, Dtype* y) {
 CUDA_KERNEL_LOOP(index, n) {
 y[index] = a[index] + b[index];
 }
 }

 template<>
 void greentea_gpu_add<float>(const int N, const float* a, const float* b,
 float* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, b, y);
 }

 template<>
 void greentea_gpu_add<double>(const int N, const double* a, const double* b,
 double* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, b, y);
 }

 template <typename Dtype>
 __global__ void sub_kernel(const int n, const Dtype* a,
 const Dtype* b, Dtype* y) {
 CUDA_KERNEL_LOOP(index, n) {
 y[index] = a[index] - b[index];
 }
 }

 template<>
 void greentea_gpu_sub<float>(const int N, const float* a, const float* b,
 float* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, b, y);
 }

 template<>
 void greentea_gpu_sub<double>(const int N, const double* a, const double* b,
 double* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, b, y);
 }

 template <typename Dtype>
 __global__ void mul_kernel(const int n, const Dtype* a,
 const Dtype* b, Dtype* y) {
 CUDA_KERNEL_LOOP(index, n) {
 y[index] = a[index] * b[index];
 }
 }

 template<>
 void greentea_gpu_mul<float>(const int N, const float* a, const float* b,
 float* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, b, y);
 }

 template<>
 void greentea_gpu_mul<double>(const int N, const double* a, const double* b,
 double* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, b, y);
 }

 template <typename Dtype>
 __global__ void div_kernel(const int n, const Dtype* a,
 const Dtype* b, Dtype* y) {
 CUDA_KERNEL_LOOP(index, n) {
 y[index] = a[index] / b[index];
 }
 }

 template<>
 void greentea_gpu_div<float>(const int N, const float* a, const float* b,
 float* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, b, y);
 }

 template<>
 void greentea_gpu_div<double>(const int N, const double* a, const double* b,
 double* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, b, y);
 }

 template <typename Dtype>
 __global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
 CUDA_KERNEL_LOOP(index, n) {
 y[index] = abs(a[index]);
 }
 }

 template<>
 void greentea_gpu_abs<float>(const int N, const float* a, float* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, y);
 }

 template<>
 void greentea_gpu_abs<double>(const int N, const double* a, double* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, y);
 }

 template <typename Dtype>
 __global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
 CUDA_KERNEL_LOOP(index, n) {
 y[index] = exp(a[index]);
 }
 }

 template<>
 void greentea_gpu_exp<float>(const int N, const float* a, float* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, y);
 }

 template<>
 void greentea_gpu_exp<double>(const int N, const double* a, double* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, y);
 }

 template <typename Dtype>
 __global__ void powx_kernel(const int n, const Dtype* a,
 const Dtype alpha, Dtype* y) {
 CUDA_KERNEL_LOOP(index, n) {
 y[index] = pow(a[index], alpha);
 }
 }

 template<>
 void greentea_gpu_powx<float>(const int N, const float* a, const float alpha,
 float* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, alpha, y);
 }

 template<>
 void greentea_gpu_powx<double>(const int N, const double* a, const double alpha,
 double* y) {
 // NOLINT_NEXT_LINE(whitespace/operators)
 powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
 N, a, alpha, y);
 }

 DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
 - (x[index] < Dtype(0)));
 DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

 __global__ void popc_kernel(const int n, const float* a, const float* b,
 uint8_t* y) {
 CUDA_KERNEL_LOOP(index, n)
 {
 y[index] = __popc(
 static_cast<uint32_t>(a[index]) ^ static_cast<uint32_t>(b[index]));
 }
 }

 __global__ void popcll_kernel(const int n, const double* a, const double* b,
 uint8_t* y) {
 CUDA_KERNEL_LOOP(index, n)
 {
 y[index] = __popcll(
 static_cast<uint64_t>(a[index]) ^ static_cast<uint64_t>(b[index]));
 }
 }

 template<>
 uint32_t greentea_gpu_hamming_distance<float>(const int n, const float* x,
 const float* y) {
 // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
 // TestHammingDistanceGPU in test_math_functions.cpp).
 NOT_IMPLEMENTED;
 thrust::device_vector<uint8_t> popcounts(n);
 // NOLINT_NEXT_LINE(whitespace/operators)
 popc_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
 n, x, y, thrust::raw_pointer_cast(popcounts.data()));
 return thrust::reduce(popcounts.begin(), popcounts.end(), (uint32_t) 0,
 thrust::plus<uint32_t>());
 }

 template<>
 uint32_t greentea_gpu_hamming_distance<double>(const int n, const double* x,
 const double* y) {
 // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
 // TestHammingDistanceGPU in test_math_functions.cpp).
 NOT_IMPLEMENTED;
 thrust::device_vector<uint8_t> popcounts(n);
 // NOLINT_NEXT_LINE(whitespace/operators)
 popcll_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
 n, x, y, thrust::raw_pointer_cast(popcounts.data()));
 return thrust::reduce(popcounts.begin(), popcounts.end(),
 (uint32_t) 0,
 thrust::plus<uint32_t>());
 }

 void greentea_gpu_rng_uniform(const int n, unsigned int* r) {
 CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
 }

 template<>
 void greentea_gpu_rng_uniform<float>(const int n, const float a, const float b,
 float* r) {
 CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
 const float range = b - a;
 if (range != static_cast<float>(1)) {
 greentea_gpu_scal(n, range, r);
 }
 if (a != static_cast<float>(0)) {
 greentea_gpu_add_scalar(n, a, r);
 }
 }

 template<>
 void greentea_gpu_rng_uniform<double>(const int n, const double a,
 const double b, double* r) {
 CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
 const double range = b - a;
 if (range != static_cast<double>(1)) {
 greentea_gpu_scal(n, range, r);
 }
 if (a != static_cast<double>(0)) {
 greentea_gpu_add_scalar(n, a, r);
 }
 }

 template<>
 void greentea_gpu_rng_gaussian(const int n, const float mu, const float sigma,
 float* r) {
 CURAND_CHECK(curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
 }

 template<>
 void greentea_gpu_rng_gaussian(const int n, const double mu, const double sigma,
 double* r) {
 CURAND_CHECK(
 curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
 }
 */

}
