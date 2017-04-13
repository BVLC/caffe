#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

#define THREADS_PER_BLOCK_CSR 32

namespace caffe {

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
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
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
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
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

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
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

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
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

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
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

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
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

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
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

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

template<typename Dtype>
__device__ void caffe_gpu_csr_gemm_kernel_core(const int M, const int N,
                                               const int K, const Dtype alpha,
                                               int nzz, const Dtype* A,
                                               const int* indices,
                                               const int* ptr, const Dtype* B,
                                               const int ldb1, const int ldb2,
                                               const Dtype beta, Dtype* C,
                                               const int ldc1, const int ldc2) {
  __shared__ volatile Dtype sums[THREADS_PER_BLOCK_CSR * 2];

  for (int rowA = blockIdx.x; rowA < M; rowA += gridDim.x) {
    const int begin = ptr[rowA];
    const int end = ptr[rowA + 1];
    const int offset_c_part = rowA * ldc1;
    for (int colC = blockIdx.y; colC < N; colC += gridDim.y) {
      Dtype sum = 0.0;
      const int offset_b_part = colC * ldb2;
      for (int pos = begin + threadIdx.x; pos < end; pos +=
          THREADS_PER_BLOCK_CSR) {
        const int colA = indices[pos];
        sum += A[pos] * B[colA * ldb1 + offset_b_part];
      }
      sums[threadIdx.x] = sum;
      __syncthreads();

      /* hardcoded reduction for 32 threads */
      sums[threadIdx.x] += sums[threadIdx.x + 16];
      sums[threadIdx.x] += sums[threadIdx.x + 8];
      sums[threadIdx.x] += sums[threadIdx.x + 4];
      sums[threadIdx.x] += sums[threadIdx.x + 2];
      sums[threadIdx.x] += sums[threadIdx.x + 1];

      if (threadIdx.x == 0) {
        const int offsetC = offset_c_part + colC * ldc2;
        C[offsetC] = beta * C[offsetC] + alpha * sums[0];
      }
    }
  }
}

template<typename Dtype>
__global__ void caffe_gpu_csr_gemm_kernel(const CBLAS_TRANSPOSE TransB,
                                          const int M, const int N, const int K,
                                          const Dtype alpha, int nzz,
                                          const Dtype* A, const int* indices,
                                          const int* ptr, const Dtype* B,
                                          const Dtype beta, Dtype* C,
                                          const CBLAS_ORDER orderC) {
  if (orderC == CblasRowMajor) {
    if (TransB == CblasNoTrans) {
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, nzz, A, indices, ptr, B, N,
                                     1, beta, C, N, 1);
    } else {
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, nzz, A, indices, ptr, B, 1,
                                     K, beta, C, N, 1);
    }
  } else {
    if (TransB == CblasNoTrans) {
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, nzz, A, indices, ptr, B, N,
                                     1, beta, C, 1, M);
    } else {
      caffe_gpu_csr_gemm_kernel_core(M, N, K, alpha, nzz, A, indices, ptr, B, 1,
                                     K, beta, C, 1, M);
    }
  }
}

template<typename Dtype>
__device__ void caffe_gpu_csr_rank1_update_kernel_core(const int M, const int N,
                                                       const Dtype alpha,
                                                       const Dtype* A,
                                                       const int* indices,
                                                       const int* ptr,
                                                       const Dtype* B, int ldb,
                                                       Dtype* C, const int ldc1,
                                                       const int ldc2) {
  const int begin = ptr[0];
  const int end = ptr[1];
  for (int pos = blockIdx.x * blockDim.x + begin + threadIdx.x; pos < end;
      pos += blockDim.x * gridDim.x) {
    const Dtype valA = A[pos] * alpha;
    const int offset_part = indices[pos] * ldc1;
    for (int colC = blockIdx.y * blockDim.y + threadIdx.y; colC < N;
        colC += blockDim.y * gridDim.y) {
      const int C_offset = offset_part + colC * ldc2;
      C[C_offset] = C[C_offset] + B[colC * ldb] * valA;
    }
  }
}

// C = alpha A * B^T +  C where A and B are vectors.
// A is a sprase vector and B is a dense vector
template<typename Dtype>
__device__ void caffe_gpu_csr_rank1_update_kernel(const int M, const int N,
                                                  const Dtype alpha,
                                                  const Dtype* A,
                                                  const int* indices,
                                                  const int* ptr,
                                                  const Dtype* B, int ldb,
                                                  Dtype* C,
                                                  const CBLAS_ORDER orderC) {
  if (orderC == CblasRowMajor) {
    caffe_gpu_csr_rank1_update_kernel_core(M, N, alpha, A, indices, ptr, B, ldb,
                                           C, N, 1);
  } else {
    caffe_gpu_csr_rank1_update_kernel_core(M, N, alpha, A, indices, ptr, B, ldb,
                                           C, 1, M);
  }
}

template<typename Dtype>
__global__ void caffe_gpu_csr_rank1_update_kernel_multi(
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const int* indices, const int* ptr,
    const Dtype* B, int ldb, Dtype* C, const CBLAS_ORDER orderC) {
  if (TransB == CblasNoTrans) {
    for (int i = 0; i < K; i++) {
      caffe_gpu_csr_rank1_update_kernel(M, N, alpha, A, indices, ptr + i,
                                        B + (N * i), 1, C, orderC);
    }
  } else {
    for (int i = 0; i < K; i++) {
      caffe_gpu_csr_rank1_update_kernel(M, N, alpha, A, indices, ptr + i, B + i,
                                        K, C, orderC);
    }
  }
}

template<>
void caffe_gpu_csr_gemm<float>(const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const int M,
                               const int N, const int K, const float alpha,
                               int nzz, const float* A, const int* indices,
                               const int* ptr, const float* B, const float beta,
                               float* C, const CBLAS_ORDER orderC) {
  if (TransA == CblasNoTrans) {
    dim3 grids(M, N);
    dim3 threads(THREADS_PER_BLOCK_CSR, 1);
    caffe_gpu_csr_gemm_kernel<float><< <grids, threads>>>(TransB, M, N, K,
        alpha, nzz, A, indices, ptr, B, beta, C, orderC);
  } else {
    // scale C by beta
    if (beta != 1.0) {
      CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle() , M * N, &beta, C, 1));
    }
    const int average_nzz_per_row = nzz/K+1;
    dim3 grids((average_nzz_per_row+64-1)/64, N);
    dim3 threads(64, 1);
    caffe_gpu_csr_rank1_update_kernel_multi<float><< <grids, threads>>>(TransB,
        M, N, K,
        alpha, A, indices, ptr , B, 1, C, orderC);
  }
}

template<>
void caffe_gpu_csr_gemm<double>(const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB, const int M,
                                const int N, const int K, const double alpha,
                                int nzz, const double* A, const int* indices,
                                const int* ptr, const double* B,
                                const double beta, double* C,
                                const CBLAS_ORDER orderC) {
  if (TransA == CblasNoTrans) {
    dim3 grids(M, N);
    dim3 threads(THREADS_PER_BLOCK_CSR, 1);
    caffe_gpu_csr_gemm_kernel<double><< <grids, threads>>> (TransB, M, N, K,
        alpha, nzz, A, indices, ptr, B, beta, C, orderC);
  } else {
    // scale C by beta
    if (beta != 1.0) {
      CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle() , M * N, &beta, C, 1));
    }
    const int average_nzz_per_row = nzz/K+1;
    dim3 grids((average_nzz_per_row+64-1)/64, N);
    dim3 threads(64, 1);
    caffe_gpu_csr_rank1_update_kernel_multi<double><< <grids, threads>>>(TransB,
        M, N, K,
        alpha, A, indices, ptr , B, 1, C, orderC);
  }
}

/* Other implementation using cusparse that is very slow at least using it like this
template <>
void caffe_gpu_csr_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, int nzz, const float* A, const int* indices, const int* ptr, const float* B, const float beta,
    float* C, const CBLAS_ORDER orderC) {

  //std::cout << "M: " << M << " N: " << N << " K: " << K << " NZZ: " << nzz <<"\n"  ;

  int ldb = (TransB == CblasNoTrans) ? N : K;
  cusparseOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

  float* Bt;
  int ldb_t;

  bool reuiqre_transpose_B = (cuTransA == CUSPARSE_OPERATION_TRANSPOSE) && (cuTransB == CUSPARSE_OPERATION_TRANSPOSE);
  if (reuiqre_transpose_B){
    //we need to transpose B because this operation is not supported by cusparse (god knows why)
    ldb_t = K;
    const float zero = 0.0;
    const float one = 1.0;
    CUDA_CHECK(cudaMalloc((void**)&Bt, sizeof(float)*K*N));
    CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_T, K, N, &one, B, ldb, &zero, B, ldb, Bt, ldb_t));
  }

  int msparse = (TransA == CblasNoTrans) ? M : K;
  int ksparse = (TransA == CblasNoTrans) ? K : M;
  if (orderC == CblasRowMajor){
    float* Ct;
    CUDA_CHECK(cudaMalloc((void**)&Ct, sizeof(float)*M*N));
    const float zero = 0.0;
    const float one = 1.0;
    if (reuiqre_transpose_B){
      CUSPARSE_CHECK(cusparseScsrmm2(Caffe::cusparse_handle(), cuTransA, CUSPARSE_OPERATION_NON_TRANSPOSE, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, Bt,  ldb_t, &zero, Ct, M));
      CUDA_CHECK(cudaFree(Bt));
    }else{
      CUSPARSE_CHECK(cusparseScsrmm2(Caffe::cusparse_handle(), cuTransA, cuTransB, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, B,  ldb, &zero, Ct, M));
    }
    CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), CUBLAS_OP_T , CUBLAS_OP_N, N, M, &one, Ct, M, &beta, C, N, C, N));
    CUDA_CHECK(cudaFree(Ct));
  }else{
    //this is the default of CUSPARSE by the Matrix B is by default rowmajor
    if (reuiqre_transpose_B){
      CUSPARSE_CHECK(cusparseScsrmm2(Caffe::cusparse_handle(), cuTransA, CUSPARSE_OPERATION_NON_TRANSPOSE, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, Bt,  ldb_t, &beta, C, M));
      CUDA_CHECK(cudaFree(Bt));
    }else{
      CUSPARSE_CHECK(cusparseScsrmm2(Caffe::cusparse_handle(), cuTransA, cuTransB, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, B,  ldb, &beta, C, M));
    }
  }
}

template <>
void caffe_gpu_csr_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, int nzz, const double* A, const int* indices, const int* ptr, const double* B, const double beta,
    double* C, const CBLAS_ORDER orderC) {

  //std::cout << "M: " << M << "N: " << N << "K: " << K << "NZZ: " << nzz  ;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cusparseOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

  double* Bt;
  int ldb_t;
  bool reuiqre_transpose_B = (cuTransA == CUSPARSE_OPERATION_TRANSPOSE) && (cuTransB == CUSPARSE_OPERATION_TRANSPOSE);
  if (reuiqre_transpose_B){
    //we need to transpose B because this operation is not supported by cusparse (god knows why)
    ldb_t = K;
    const double zero = 0.0;
    const double one = 1.0;
    CUDA_CHECK(cudaMalloc((void**)&Bt, sizeof(double)*K*N));
    CUBLAS_CHECK(cublasDgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_T, K, N, &one, B, ldb, &zero, B, ldb, Bt, ldb_t));
  }

  int msparse = (TransA == CblasNoTrans) ? M : K;
  int ksparse = (TransA == CblasNoTrans) ? K : M;
  if (orderC == CblasRowMajor){
    double* Ct;
    CUDA_CHECK(cudaMalloc((void**)&Ct, sizeof(double)*M*N));
    const double zero = 0.0;
    const double one = 1.0;
    if (reuiqre_transpose_B){
      CUSPARSE_CHECK(cusparseDcsrmm2(Caffe::cusparse_handle(), cuTransA, CUSPARSE_OPERATION_NON_TRANSPOSE, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, Bt,  ldb_t, &zero, Ct, M));
      CUDA_CHECK(cudaFree(Bt));
    }else{
      CUSPARSE_CHECK(cusparseDcsrmm2(Caffe::cusparse_handle(), cuTransA, cuTransB, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, B,  ldb, &zero, Ct, M));
    }
    CUBLAS_CHECK(cublasDgeam(Caffe::cublas_handle(), CUBLAS_OP_T , CUBLAS_OP_N, N, M, &one, Ct, M, &beta, C, N, C, N));
    CUDA_CHECK(cudaFree(Ct));
  }else{
    //this is the default of CUSPARSE by the Matrix B is by default rowmajor
    if (reuiqre_transpose_B){
      CUSPARSE_CHECK(cusparseDcsrmm2(Caffe::cusparse_handle(), cuTransA, CUSPARSE_OPERATION_NON_TRANSPOSE, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, Bt,  ldb_t, &beta, C, M));
      CUDA_CHECK(cudaFree(Bt));
    }else{
      CUSPARSE_CHECK(cusparseDcsrmm2(Caffe::cusparse_handle(), cuTransA, cuTransB, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, B,  ldb, &beta, C, M));
    }
  }
}

*/

}  // namespace caffe

