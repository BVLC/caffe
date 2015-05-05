/*
 * greentea_math_functions.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */
#ifdef USE_GREENTEA
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

#ifdef USE_CLBLAS
#include <clBLAS.h>
#endif

#ifdef USE_VIENNACLBLAS
#include "libviennacl/include/viennacl.hpp"
#endif

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

template<typename Dtype>
void greentea_gpu_gemm(const int ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int M, const int N,
                       const int K, const Dtype alpha, const cl_mem A,
                       const int offA, const cl_mem B, const int offB,
                       const Dtype beta, cl_mem C, const int offC) {

  int offArow = offA;
  int offBrow = offB;
  int offCrow = offC;

  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;

#ifdef USE_VIENNACLBLAS

  int offAcol = 0;
  int incArow = 1;
  int incAcol = 1;
  int offBcol = 0;
  int incBrow = 1;
  int incBcol = 1;
  int offCcol = 0;
  int incCrow = 1;
  int incCcol = 1;

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
    GREENTEA_VCL_BLAS_CHECK(
        ViennaCLOpenCLSgemm(backend, vclOrderA, vclTransA, vclOrderB, vclTransB,
                            vclOrderC, M, N, K, alpha, A, offArow, offAcol,
                            incArow, incAcol, lda, B, offBrow, offBcol, incBrow,
                            incBcol, ldb, beta, C, offCrow, offCcol, incCrow,
                            incCcol, ldc));
  } else {
    GREENTEA_VCL_BLAS_CHECK(
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
    GREENTEA_CL_BLAS_CHECK(
        clblasSgemm(clOrder, clTransA, clTransB, M, N, K, alpha, A, offArow, lda, B, offBrow, ldb, beta, C, offCrow, ldc, 1, &queue, 0, NULL, NULL));
  } else {
    GREENTEA_CL_BLAS_CHECK(
        clblasDgemm(clOrder, clTransA, clTransB, M, N, K, alpha, A, offArow, lda, B, offBrow, ldb, beta, C, offCrow, ldc, 1, &queue, 0, NULL, NULL));
  }
#endif

}

template void greentea_gpu_gemm<float>(const int ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int M, const int N, const int K,
                                       const float alpha, const cl_mem A,
                                       const int offA, const cl_mem B,
                                       const int offB, const float beta,
                                       cl_mem C, const int offC);
template void greentea_gpu_gemm<double>(const int ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const CBLAS_TRANSPOSE TransB,
                                        const int M, const int N, const int K,
                                        const double alpha, const cl_mem A,
                                        const int offA, const cl_mem B,
                                        const int offB, const double beta,
                                        cl_mem C, const int offC);

template<typename Dtype>
void greentea_gpu_gemv(const int ctx_id, const CBLAS_TRANSPOSE TransA,
                       const int M, const int N, const Dtype alpha,
                       const cl_mem A, const int offA, const cl_mem x,
                       const int offx, const Dtype beta, cl_mem y,
                       const int offy) {

  int lda = (TransA == CblasNoTrans) ? N : M;

#ifdef USE_VIENNACLBLAS
  // TODO
#endif

#ifdef USE_CLBLAS

  clblasOrder clOrder = clblasRowMajor;
  clblasTranspose clTransA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;

  viennacl::ocl::context ctx = viennacl::ocl::get_context(ctx_id);
  cl_command_queue queue = ctx.get_queue().handle().get();

  if (std::is_same<Dtype, float>::value) {
    GREENTEA_CL_BLAS_CHECK(
        clblasSgemv(clOrder,clTransA,M,N,alpha,A,offA,lda,x,offx,1,beta,y,offy,1,1,&queue,0,NULL,NULL));
  } else {
    GREENTEA_CL_BLAS_CHECK(
        clblasDgemv(clOrder,clTransA,M,N,alpha,A,offA,lda,x,offx,1,beta,y,offy,1,1,&queue,0,NULL,NULL));
  }
#endif
}

template void greentea_gpu_gemv<float>(const int ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const int M, const int N,
                                       const float alpha, const cl_mem A,
                                       const int offA, const cl_mem x,
                                       const int offx, const float beta,
                                       cl_mem y, const int offy);
template void greentea_gpu_gemv<double>(const int ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const int M, const int N,
                                        const double alpha, const cl_mem A,
                                        const int offA, const cl_mem x,
                                        const int offx, const double beta,
                                        cl_mem y, const int offy);

template<typename Dtype>
void greentea_gpu_axpy(const int ctx_id, const int N, const Dtype alpha,
                       const cl_mem X, const int offX, cl_mem Y,
                       const int offY) {

#ifdef USE_VIENNACLBLAS
  // TODO
#endif

#ifdef USE_CLBLAS
  viennacl::ocl::context ctx = viennacl::ocl::get_context(ctx_id);
  cl_command_queue queue = ctx.get_queue().handle().get();

  if (std::is_same<Dtype, float>::value) {
    GREENTEA_CL_BLAS_CHECK(
        clblasSaxpy(N,alpha,X,offX,1,Y,offY,1,1,&queue,0,NULL,NULL));
  } else {
    GREENTEA_CL_BLAS_CHECK(
        clblasDaxpy(N,alpha,X,offX,1,Y,offY,1,1,&queue,0,NULL,NULL));

  }
#endif
}

template void greentea_gpu_axpy<float>(const int ctx_id, const int N,
                                       const float alpha, const cl_mem X,
                                       const int offX, cl_mem Y,
                                       const int offY);
template void greentea_gpu_axpy<double>(const int ctx_id, const int N,
                                        const double alpha, const cl_mem X,
                                        const int offX, cl_mem Y,
                                        const int offY);

template<typename Dtype>
void greentea_gpu_mul(const int ctx_id, const int N, const cl_mem a,
                      const int offa, const cl_mem b, const int offb, cl_mem y,
                      const int offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_mul = program.get_kernel(
      CL_KERNEL_SELECT("kernel_mul"));
  viennacl::ocl::enqueue(
      oclk_mul(N, WrapHandle(a, ctx), offa, WrapHandle(b, ctx), offb,
               WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_mul<float>(const int ctx_id, const int N,
                                      const cl_mem a, const int offa,
                                      const cl_mem b, const int offb, cl_mem y,
                                      const int offy);
template void greentea_gpu_mul<double>(const int ctx_id, const int N,
                                       const cl_mem a, const int offa,
                                       const cl_mem b, const int offb, cl_mem y,
                                       const int offy);

/*
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
#endif
