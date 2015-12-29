#ifdef USE_OCL
#include <clBLAS.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

using std::pair;

namespace caffe {

#ifndef ulong
  typedef unsigned long ulong;
#endif

void caffe_gpu_fill_buffer_size_t(int N, int alpha, void* X) {
  ClState& state = Caffe::cl_state();
  ClKernel kernel = state.get_kernel("caffe_Fill_Buffer");
  char alphaChar = alpha;
  kernel.set_arg(0, N);
  kernel.set_arg(1, alphaChar);
  kernel.set_arg(2, X);
  kernel.enqueue_blocking(N);
}


template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // GEMM: A[MxK], B[KxN], C[MxN]
  // C = (alpha A B) + (beta C)
  // C = (alpha A^T B) + (beta C)
  // C = (alpha A B^T) + (beta C)
  // C = (alpha A^T B^T) + (beta C)
  // Note that Caffe uses row-major order so use row-major order for clBLAS.
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufA = clState.get_buffer_mem(A);
  ClMemOff<float> bufB = clState.get_buffer_mem(B);
  ClMemOff<float> bufC = clState.get_buffer_mem(C);

  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;

  cl_command_queue queue = clState.get_command_queue();
  clblasTranspose clTransA = (TransA == CblasNoTrans) ? clblasNoTrans :
      clblasTrans;
  clblasTranspose clTransB = (TransB == CblasNoTrans) ? clblasNoTrans :
      clblasTrans;

  OCL_CHECK(clblasSgemm(clblasRowMajor, clTransA, clTransB, M, N, K, alpha,
      bufA.memobj, bufA.offset, lda,
      bufB.memobj, bufB.offset, ldb, beta,
      bufC.memobj, bufC.offset, ldc, 1, &queue, 0, NULL, NULL));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // GEMM: A[MxK], B[KxN], C[MxN]
  // C = (alpha A B) + (beta C)
  // C = (alpha A^T B) + (beta C)
  // C = (alpha A B^T) + (beta C)
  // C = (alpha A^T B^T) + (beta C)
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufA = clState.get_buffer_mem(A);
  ClMemOff<double> bufB = clState.get_buffer_mem(B);
  ClMemOff<double> bufC = clState.get_buffer_mem(C);

  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;

  cl_command_queue queue = clState.get_command_queue();
  clblasTranspose clTransA = (TransA == CblasNoTrans) ? clblasNoTrans :
      clblasTrans;
  clblasTranspose clTransB = (TransB == CblasNoTrans) ? clblasNoTrans :
      clblasTrans;

  OCL_CHECK(clblasDgemm(clblasRowMajor, clTransA, clTransB, M, N, K, alpha,
      bufA.memobj, bufA.offset, lda,
      bufB.memobj, bufB.offset, ldb, beta,
      bufC.memobj, bufC.offset, ldc, 1, &queue, 0, NULL, NULL));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  // GEMV: A[MxN], x[N], y[M]
  // y = (alpha A x) + (beta y)
  // y = (alpha A^T x) + (beta y)
  // Note that Caffe uses row-major order so use row-major order for clBLAS.
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufA = clState.get_buffer_mem(A);
  ClMemOff<float> bufx = clState.get_buffer_mem(x);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);
  cl_command_queue queue = clState.get_command_queue();
  clblasTranspose clTransA = (TransA == CblasNoTrans) ? clblasNoTrans :
      clblasTrans;
  OCL_CHECK(clblasSgemv(clblasRowMajor, clTransA, M, N, alpha,
      bufA.memobj, bufA.offset, N,
      bufx.memobj, bufx.offset, 1, beta,
      bufy.memobj, bufy.offset, 1, 1, &queue, 0, NULL, NULL));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  // GEMV: A[MxN], x[N], y[M]
  // y = (alpha A x) + (beta y)
  // y = (alpha A^T x) + (beta y)
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufA = clState.get_buffer_mem(A);
  ClMemOff<double> bufx = clState.get_buffer_mem(x);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);
  cl_command_queue queue = clState.get_command_queue();
  clblasTranspose clTransA = (TransA == CblasNoTrans) ? clblasNoTrans :
      clblasTrans;
  OCL_CHECK(clblasDgemv(clblasRowMajor, clTransA, M, N, alpha,
      bufA.memobj, bufA.offset, N,
      bufx.memobj, bufx.offset, 1, beta,
      bufy.memobj, bufy.offset, 1, 1, &queue, 0, NULL, NULL));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  // AXPY: X[N], Y[N]
  // Y = alpha X + Y
  ClState& state = Caffe::cl_state();
  ClMemOff<float> bufX = state.get_buffer_mem(X);
  ClMemOff<float> bufY = state.get_buffer_mem(Y);
  cl_command_queue queue = state.get_command_queue();
  OCL_CHECK(clblasSaxpy(N, alpha, bufX.memobj, bufX.offset, 1, bufY.memobj,
      bufY.offset, 1, 1, &queue, 0, NULL, NULL));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  // AXPY: X[N], Y[N]
  // Y = alpha X + Y
  ClState& state = Caffe::cl_state();
  ClMemOff<double> bufX = state.get_buffer_mem(X);
  ClMemOff<double> bufY = state.get_buffer_mem(Y);
  cl_command_queue queue = state.get_command_queue();
  OCL_CHECK(clblasDaxpy(N, alpha, bufX.memobj, bufX.offset, 1, bufY.memobj,
      bufY.offset, 1, 1, &queue, 0, NULL, NULL));
}

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y) {
  if (X == Y) return;

  ClState& state = Caffe::cl_state();
  cl_command_queue queue = state.get_command_queue();

  ClMemOff<uint8_t> bufX = state.get_buffer_mem(X);
  ClMemOff<uint8_t> bufY = state.get_buffer_mem(Y);

  if (bufX.memobj != NULL && bufY.memobj != NULL) {
    OCL_CHECK(clEnqueueCopyBuffer(queue, bufX.memobj, bufY.memobj, bufX.offset,
        bufY.offset, N, 0, NULL, NULL));
  } else if (bufX.memobj != NULL) {
    OCL_CHECK(clEnqueueReadBuffer(queue, bufX.memobj, CL_TRUE, bufX.offset, N,
        Y, 0, NULL, NULL));
  } else if (bufY.memobj != NULL) {
    OCL_CHECK(clEnqueueWriteBuffer(queue, bufY.memobj, CL_TRUE, bufY.offset, N,
        X, 0, NULL, NULL));
  } else {
    memcpy(Y, X, N);
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  // SCAL: X[N]
  // X = alpha X
  ClState& state = Caffe::cl_state();
  ClMemOff<float> bufX = state.get_buffer_mem(X);
  cl_command_queue queue = state.get_command_queue();
  OCL_CHECK(clblasSscal(N, alpha, bufX.memobj, bufX.offset, 1, 1, &queue,
      0, NULL, NULL));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  // SCAL: X[N]
  // X = alpha X
  ClState& state = Caffe::cl_state();
  ClMemOff<double> bufX = state.get_buffer_mem(X);
  cl_command_queue queue = state.get_command_queue();
  OCL_CHECK(clblasDscal(N, alpha, bufX.memobj, bufX.offset, 1, 1, &queue,
      0, NULL, NULL));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  // SCAL
  caffe_gpu_scal<float>(N, beta, Y);
  // AXPY
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  // SCAL
  caffe_gpu_scal<double>(N, beta, Y);
  // AXPY
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  // DOT: x[n], y[n]
  // out (host memory) = x dot y
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufx = clState.get_buffer_mem(x);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);
  float* memDotP = static_cast<float*>(clState.create_buffer(
      CL_MEM_WRITE_ONLY, sizeof(cl_float), NULL));
  float* memScratch = static_cast<float*>(clState.create_buffer(
      CL_MEM_READ_WRITE, n * sizeof(cl_float), NULL));
  ClMemOff<float> bufDotP = clState.get_buffer_mem(memDotP);
  ClMemOff<float> bufScratch = clState.get_buffer_mem(memScratch);
  cl_command_queue queue = clState.get_command_queue();
  OCL_CHECK(clblasSdot(n, bufDotP.memobj, 0, bufx.memobj, bufx.offset, 1,
      bufy.memobj, bufy.offset, 1, bufScratch.memobj,
      1, &queue, 0, NULL, NULL));
  OCL_CHECK(clEnqueueReadBuffer(queue, bufDotP.memobj, CL_TRUE, 0,
      sizeof(cl_float), out, 0, NULL, NULL));

  clState.destroy_buffer(memScratch);
  clState.destroy_buffer(memDotP);
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  // DOT: x[n], y[n]
  // out = x dot y
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufx = clState.get_buffer_mem(x);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);
  double* memDotP = static_cast<double*>(clState.create_buffer(
      CL_MEM_WRITE_ONLY, sizeof(cl_double), NULL));
  double* memScratch = static_cast<double*>(clState.create_buffer(
      CL_MEM_READ_WRITE, n * sizeof(cl_double), NULL));
  ClMemOff<double> bufDotP = clState.get_buffer_mem(memDotP);
  ClMemOff<double> bufScratch = clState.get_buffer_mem(memScratch);
  cl_command_queue queue = clState.get_command_queue();
  OCL_CHECK(clblasDdot(n, bufDotP.memobj, 0, bufx.memobj, bufx.offset, 1,
      bufy.memobj, bufy.offset, 1, bufScratch.memobj, 1, &queue, 0, NULL,
      NULL));
  OCL_CHECK(clEnqueueReadBuffer(queue, bufDotP.memobj, CL_TRUE, 0,
      sizeof(cl_double), out, 0, NULL, NULL));
  clState.destroy_buffer(memScratch);
  clState.destroy_buffer(memDotP);
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  // ASUM: x[n]
  // y = absolute sum of values in x, y is a host memory
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufx = clState.get_buffer_mem(x);
  float* memAsum = static_cast<float*>(clState.create_buffer(
      CL_MEM_WRITE_ONLY, sizeof(cl_float), NULL));
  float* memScratch = static_cast<float*>(clState.create_buffer(
      CL_MEM_READ_WRITE, n * sizeof(cl_float), NULL));
  ClMemOff<float> bufAsum = clState.get_buffer_mem(memAsum);
  ClMemOff<float> bufScratch = clState.get_buffer_mem(memScratch);
  cl_command_queue queue = clState.get_command_queue();
  OCL_CHECK(clblasSasum(n, bufAsum.memobj, 0, bufx.memobj, bufx.offset, 1,
      bufScratch.memobj, 1, &queue, 0, NULL, NULL));
  OCL_CHECK(clEnqueueReadBuffer(queue, bufAsum.memobj, CL_TRUE, 0,
      sizeof(cl_float), y, 0, NULL, NULL));
  clState.destroy_buffer(memScratch);
  clState.destroy_buffer(memAsum);
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  // ASUM: x[n]
  // y = absolute sum of values in x
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufx = clState.get_buffer_mem(x);
  double* memAsum = static_cast<double*>(clState.create_buffer(
      CL_MEM_WRITE_ONLY, sizeof(cl_double), NULL));
  double* memScratch = static_cast<double*>(clState.create_buffer(
      CL_MEM_READ_WRITE, n * sizeof(cl_double), NULL));
  ClMemOff<double> bufAsum = clState.get_buffer_mem(memAsum);
  ClMemOff<double> bufScratch = clState.get_buffer_mem(memScratch);
  cl_command_queue queue = clState.get_command_queue();
  OCL_CHECK(clblasDasum(n, bufAsum.memobj, 0, bufx.memobj, bufx.offset, 1,
      bufScratch.memobj, 1, &queue, 0, NULL, NULL));
  OCL_CHECK(clEnqueueReadBuffer(queue, bufAsum.memobj, CL_TRUE, 0,
      sizeof(cl_double), y, 0, NULL, NULL));
  clState.destroy_buffer(memScratch);
  clState.destroy_buffer(memAsum);
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
    float* y) {
  // SCALE: x[n], y[n]
  // y = alpha x => COPY followed by SCAL
  // COPY: x to y
  ClState& state = Caffe::cl_state();
  ClMemOff<float> bufx = state.get_buffer_mem(x);
  ClMemOff<float> bufy = state.get_buffer_mem(y);
  cl_int err;
  cl_command_queue queue = state.get_command_queue();
  OCL_CHECK(err = clblasScopy(n, bufx.memobj, bufx.offset, 1, bufy.memobj,
      bufy.offset, 1, 1, &queue, 0, NULL, NULL));
  // SCAL: y = alpha y
  OCL_CHECK(clblasSscal(n, alpha, bufy.memobj, bufy.offset, 1, 1, &queue, 0,
      NULL, NULL));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double* x,
    double* y) {
  // SCALE: x[n], y[n]
  // y = alpha x => COPY followed by SCAL
  // COPY: x to y
  ClState& state = Caffe::cl_state();
  ClMemOff<double> bufx = state.get_buffer_mem(x);
  ClMemOff<double> bufy = state.get_buffer_mem(y);
  cl_int err;
  cl_command_queue queue = state.get_command_queue();
  OCL_CHECK(err = clblasDcopy(n, bufx.memobj, bufx.offset, 1, bufy.memobj,
      bufy.offset, 1, 1, &queue, 0, NULL, NULL));
  // SCAL: y = alpha y
  OCL_CHECK(clblasDscal(n, alpha, bufy.memobj, bufy.offset, 1, 1, &queue,
      0, NULL, NULL));
}

template <>
void caffe_gpu_set<int>(const int N, const int alpha, int* Y) {
  ClDeviceProperties cl_prop_version = Caffe::cl_state().get_properties();
  if (cl_prop_version.version == "OpenCL 1.1") {
    ClState& state = Caffe::cl_state();
    ClKernel kernel = state.get_kernel("caffe_Fill_Buffer");
    kernel.set_arg(0, N);
    kernel.set_arg(1, alpha);
    kernel.set_arg(2, Y);
    kernel.enqueue_blocking(N);
  } else {
    caffe_gpu_memset(N*sizeof(alpha), alpha, Y);
  }
}

template <>
void caffe_gpu_set<float>(const int N, const float alpha, float* Y) {
  ClState& state = Caffe::cl_state();
  ClDeviceProperties cl_prop_version = Caffe::cl_state().get_properties();

  if (cl_prop_version.version == "OpenCL 1.1") {
    ClKernel kernel = state.get_kernel("caffe_Fill_Buffer");
    kernel.set_arg(0, N);
    kernel.set_arg(1, alpha);
    kernel.set_arg(2, Y);
    kernel.enqueue_blocking(N);
  } else {
    ClMemOff<uint8_t> bufY = state.get_buffer_mem(static_cast<void*>(Y));
    OCL_CHECK(clEnqueueFillBuffer(state.get_command_queue(), bufY.memobj,
        &alpha, sizeof alpha, bufY.offset, N * sizeof alpha, 0, NULL, NULL));
  }
}

template <>
void caffe_gpu_set<double>(const int N, const double alpha, double* Y) {
  ClState& state = Caffe::cl_state();
  ClDeviceProperties cl_prop_version = Caffe::cl_state().get_properties();
  if (cl_prop_version.version == "OpenCL 1.1") {
    ClKernel kernel = state.get_kernel("caffe_Fill_Buffer");
    kernel.set_arg(0, N);
    kernel.set_arg(1, alpha);
    kernel.set_arg(2, Y);
    kernel.enqueue_blocking(N);
  } else {
    ClMemOff<uint8_t> bufY = state.get_buffer_mem(static_cast<void*>(Y));
    OCL_CHECK(clEnqueueFillBuffer(state.get_command_queue(), bufY.memobj,
        &alpha, sizeof alpha, bufY.offset, N * sizeof alpha, 0, NULL, NULL));
  }
}

template <>
void caffe_gpu_add_scalar<float>(const int N, const float alpha, float* Y) {
  Caffe::cl_state().get_kernel("Tadd_scalar").enqueue_params(N, N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar<double>(const int N, const double alpha, double* Y) {
  Caffe::cl_state().get_kernel("Tadd_scalar").enqueue_params(N, N, alpha, Y);
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  Caffe::cl_state().get_kernel("Tadd").enqueue_params(N, N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  Caffe::cl_state().get_kernel("Tadd").enqueue_params(N, N, a, b, y);
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  Caffe::cl_state().get_kernel("Tsub").enqueue_params(N, N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  Caffe::cl_state().get_kernel("Tsub").enqueue_params(N, N, a, b, y);
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a, const float* b,
    float* y) {
  Caffe::cl_state().get_kernel("Tmul").enqueue_params(N, N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a, const double* b,
    double* y) {
  Caffe::cl_state().get_kernel("Tmul").enqueue_params(N, N, a, b, y);
}

template <>
void caffe_gpu_div<float>(const int N, const float* a, const float* b,
    float* y) {
  Caffe::cl_state().get_kernel("Tdiv").enqueue_params(N, N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a, const double* b,
    double* y) {
  Caffe::cl_state().get_kernel("Tdiv").enqueue_params(N, N, a, b, y);
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  Caffe::cl_state().get_kernel("Tabs").enqueue_params(N, N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  Caffe::cl_state().get_kernel("Tabs").enqueue_params(N, N, a, y);
}

template <>
void caffe_gpu_exp<float>(const int n, const float* a, float* y) {
  Caffe::cl_state().get_kernel("Texp").enqueue_params(n, n, a, y);
}

template <>
void caffe_gpu_exp<double>(const int n, const double* a, double* y) {
  Caffe::cl_state().get_kernel("Texp").enqueue_params(n, n, a, y);
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  Caffe::cl_state().get_kernel("Tlog").enqueue_params(N, N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  Caffe::cl_state().get_kernel("Tlog").enqueue_params(N, N, a, y);
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a, const float alpha,
    float* y) {
  ClKernel kernel = Caffe::cl_state().get_kernel("Tpowx");
  kernel.set_arg(0, N);
  kernel.set_arg_ptr_off(1, a);
  kernel.set_arg(3, alpha);
  kernel.set_arg_ptr_off(4, y);
  kernel.enqueue(N);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  ClKernel kernel = Caffe::cl_state().get_kernel("Tpowx");
  kernel.set_arg(0, N);
  kernel.set_arg_ptr_off(1, a);
  kernel.set_arg(3, alpha);
  kernel.set_arg_ptr_off(4, y);
  kernel.enqueue(N);
}

template <>
void caffe_gpu_sign<float>(const int n, const float* x, float* y) {
  Caffe::cl_state().get_kernel("Tsign").enqueue_params(n, n, x, y);
}

template <>
void caffe_gpu_sign<double>(const int n, const double* x, double* y) {
  Caffe::cl_state().get_kernel("Tsign").enqueue_params(n, n, x, y);
}

template <>
void caffe_gpu_sgnbit<float>(const int n, const float* x, float* y) {
  Caffe::cl_state().get_kernel("Tsignbit").enqueue_params(n, n, x, y);
}

template <>
void caffe_gpu_sgnbit<double>(const int n, const double* x, double* y) {
  Caffe::cl_state().get_kernel("Tsignbit").enqueue_params(n, n, x, y);
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  // Generate uniform n random numbers and put them into a device memory r
  ClState& state = Caffe::cl_state();
  ClKernel kernel = state.get_kernel("R123_rng_uint32");

  ulong seed[3] = {caffe_rng_rand(), caffe_rng_rand(), caffe_rng_rand()};
  void* memSeed = state.create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof seed, seed);

  kernel.set_arg(0, memSeed);
  kernel.set_arg(1, n);
  kernel.set_arg(2, r);
  kernel.enqueue_blocking(n);

  state.destroy_buffer(memSeed);
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  // Generate uniform n random numbers and put them into a device memory r
  ClState& state = Caffe::cl_state();
  ClKernel kernel = state.get_kernel("R123_rng_flt32");

  ulong seed[3] = {caffe_rng_rand(), caffe_rng_rand(), caffe_rng_rand()};
  void* memSeed = state.create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof seed, seed);

  kernel.set_arg(0, memSeed);
  kernel.set_arg(1, n);
  kernel.set_arg(2, r);
  kernel.enqueue_blocking(n);

  state.destroy_buffer(memSeed);

  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal<float>(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar<float>(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  NOT_IMPLEMENTED;
}


template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  ClState& state = Caffe::cl_state();
  ClKernel kernel = state.get_kernel("rng_normal_flt32");

  ulong seed[3] = {caffe_rng_rand(), caffe_rng_rand(), caffe_rng_rand()};
  void* memSeed = state.create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof seed, seed);

  kernel.set_arg(0, memSeed);
  kernel.set_arg(1, n);
  kernel.set_arg(2, mu);
  kernel.set_arg(3, sigma);
  kernel.set_arg(4, r);
  kernel.enqueue_blocking(n);

  state.destroy_buffer(memSeed);
}


template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  NOT_IMPLEMENTED;
}

}  // namespace caffe
#endif  // USE_OCL
