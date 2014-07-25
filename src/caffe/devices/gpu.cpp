// Copyright 2014 BVLC and contributors.

#include <cublas_v2.h>

#include "caffe/common.hpp"
#include "caffe/devices/gpu.hpp"

namespace caffe {

template<>
void GPUDevice<float>::gemm(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB, const int M,
                            const int N, const int K, const float alpha,
                            const float* A, const float* B, const float beta,
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

template<>
void GPUDevice<double>::gemm(const CBLAS_TRANSPOSE TransA,
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
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template<>
void GPUDevice<int>::gemm(const CBLAS_TRANSPOSE TransA,
                          const CBLAS_TRANSPOSE TransB, const int M,
                          const int N, const int K, const int alpha,
                          const int* A, const int* B, const int beta, int* C) {
  NOT_IMPLEMENTED;
}

template<>
void GPUDevice<unsigned int>::gemm(const CBLAS_TRANSPOSE TransA,
                                   const CBLAS_TRANSPOSE TransB, const int M,
                                   const int N, const int K,
                                   const unsigned int alpha,
                                   const unsigned int* A, const unsigned int* B,
                                   const unsigned int beta, unsigned int* C) {
  NOT_IMPLEMENTED;
}

template<>
void GPUDevice<float>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                            const int N, const float alpha, const float* A,
                            const float* x, const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template<>
void GPUDevice<double>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                             const int N, const double alpha, const double* A,
                             const double* x, const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template<>
void GPUDevice<int>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                          const int N, const int alpha, const int* A,
                          const int* x, const int beta, int* y) {
  NOT_IMPLEMENTED;
}

template<>
void GPUDevice<unsigned int>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                                   const int N, const unsigned int alpha,
                                   const unsigned int* A, const unsigned int* x,
                                   const unsigned int beta, unsigned int* y) {
  NOT_IMPLEMENTED;
}

template<>
void GPUDevice<float>::axpy(const int N, const float alpha, const float* X,
                            float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template<>
void GPUDevice<double>::axpy(const int N, const double alpha, const double* X,
                             double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template<>
void GPUDevice<int>::axpy(const int N, const int alpha, const int* X, int* Y) {
  NOT_IMPLEMENTED;
}

template<>
void GPUDevice<unsigned int>::axpy(const int N, const unsigned int alpha,
                                   const unsigned int* X, unsigned int* Y) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUDevice<Dtype>::axpby(const int N, const Dtype alpha,
                             const Dtype* X, const Dtype beta, Dtype* Y) {
  this->scal(N, beta, Y);
  this->axpy(N, alpha, X, Y);
}

template<typename Dtype>
/* NOLINT_NEXT_LINE(build/include_what_you_use) */
void GPUDevice<Dtype>::copy(const int N, const Dtype *X, Dtype *Y) {
  CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
}

template<>
void GPUDevice<float>::scal(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template<>
void GPUDevice<double>::scal(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template<>
void GPUDevice<int>::scal(const int N, const int alpha, int *X) {
  NOT_IMPLEMENTED;
}

template<>
void GPUDevice<unsigned int>::scal(const int N, const unsigned int alpha,
                                   unsigned int *X) { NOT_IMPLEMENTED; }

template<>
void GPUDevice<float>::dot(const int N, const float* x, const float* y,
                           float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), N, x, 1, y, 1, out));
}

template<>
void GPUDevice<double>::dot(const int N, const double* x, const double* y,
                            double* out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), N, x, 1, y, 1, out));
}

template<>
void GPUDevice<int>::dot(const int N, const int* x, const int* y, int* out) {
  NOT_IMPLEMENTED;
}

template<>
void GPUDevice<unsigned int>::dot(const int N, const unsigned int* x,
                                  const unsigned int* y, unsigned int* out) {
  NOT_IMPLEMENTED;
}

// Returns the sum of the absolute values of the elements of vector x
template<>
void GPUDevice<float>::asum(const int N, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), N, x, 1, y));
}

template<>
void GPUDevice<double>::asum(const int N, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), N, x, 1, y));
}

template<>
void GPUDevice<int>::asum(const int N, const int* x, int* y) {
  NOT_IMPLEMENTED;
}

template<>
void GPUDevice<unsigned int>::asum(const int N, const unsigned int* x,
                                   unsigned int* y) { NOT_IMPLEMENTED; }

template<typename Dtype>
void GPUDevice<Dtype>::scale(const int N, const Dtype alpha, const Dtype *x,
                             Dtype* y) {
  this->copy(N, x, y);
  this->scal(N, alpha, y);
}

INSTANTIATE_CLASS(GPUDevice);
template class GPUDevice<int>;
template class GPUDevice<unsigned int>;

}  // namespace caffe
