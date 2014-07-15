// Copyright 2014 BVLC and contributors.

#include <cublas_v2.h>

#include "caffe/common.hpp"
#include "caffe/device.hpp"

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
void GPUDevice<float>::axpy(const int N, const float alpha, const float* X,
                            float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template<>
void GPUDevice<double>::axpy(const int N, const double alpha, const double* X,
                             double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template<typename Dtype>
void GPUDevice<Dtype>::axpby(const int N, const Dtype alpha,
                             const Dtype* X, const Dtype beta, Dtype* Y) {
  this->scal(N, beta, Y);
  this->axpy(N, alpha, X, Y);
}

template<>
/* NOLINT_NEXT_LINE(build/include_what_you_use) */
void GPUDevice<float>::copy(const int N, const float *X, float *Y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), N, X, 1, Y, 1));
}

template<>
/* NOLINT_NEXT_LINE(build/include_what_you_use) */
void GPUDevice<double>::copy(const int N, const double *X, double *Y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), N, X, 1, Y, 1));
}

template<typename Dtype>
void GPUDevice<Dtype>::copy_from_cpu(const int N, const Dtype *X, Dtype *Y) {
  CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyHostToDevice));
}

template<typename Dtype>
void GPUDevice<Dtype>::set(const int N, const Dtype alpha, Dtype *X) {
  caffe_gpu_set<Dtype>(N, alpha, X);
}

template<typename Dtype>
void GPUDevice<Dtype>::add_scalar(const int N, const Dtype alpha,
                                       Dtype *X) {
  caffe_gpu_add_scalar<Dtype>(N, alpha, X);
}

template<>
void GPUDevice<float>::scal(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template<>
void GPUDevice<double>::scal(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template<typename Dtype>
void GPUDevice<Dtype>::sqr(const int N, const Dtype* a, Dtype* y) {
  // TODO: implement this
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUDevice<Dtype>::add(const int N, const Dtype* a, const Dtype* b,
                                Dtype* y) {
  caffe_gpu_add<Dtype>(N, a, b, y);
}

template<typename Dtype>
void GPUDevice<Dtype>::sub(const int N, const Dtype* a, const Dtype* b,
                                Dtype* y) {
  caffe_gpu_sub<Dtype>(N, a, b, y);
}

template<typename Dtype>
void GPUDevice<Dtype>::mul(const int N, const Dtype* a, const Dtype* b,
                                Dtype* y) {
  caffe_gpu_mul<Dtype>(N, a, b, y);
}

template<typename Dtype>
void GPUDevice<Dtype>::div(const int N, const Dtype* a, const Dtype* b,
                                Dtype* y) {
  caffe_gpu_div<Dtype>(N, a, b, y);
}

template<typename Dtype>
void GPUDevice<Dtype>::powx(const int N, const Dtype* a, const Dtype b,
                                 Dtype* y) {
  caffe_gpu_powx<Dtype>(N, a, b, y);
}

template<typename Dtype>
void GPUDevice<Dtype>::rng_uniform(const int N, const Dtype a,
                                        const Dtype b, Dtype* r) {
  caffe_gpu_rng_uniform<Dtype>(N, a, b, r);
}

template<typename Dtype>
void GPUDevice<Dtype>::rng_gaussian(const int N, const Dtype mu,
                                         const Dtype sigma, Dtype* r) {
  caffe_gpu_rng_gaussian<Dtype>(N, mu, sigma, r);
}

template<typename Dtype>
void GPUDevice<Dtype>::rng_bernoulli(const int N, const Dtype p, int* r) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void GPUDevice<Dtype>::exp(const int N, const Dtype* a, Dtype* y) {
  // TODO: implement this
  NOT_IMPLEMENTED;
}

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

template<typename Dtype>
void GPUDevice<Dtype>::hamming_distance(const int N, const Dtype* x,
                                        const Dtype* y, int* out) {
  *out = caffe_gpu_hamming_distance<Dtype>(N, x, y);
}

template<>
// Returns the sum of the absolute values of the elements of vector x
void GPUDevice<float>::asum(const int N, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), N, x, 1, y));
}

template<>
// Returns the sum of the absolute values of the elements of vector x
void GPUDevice<double>::asum(const int N, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), N, x, 1, y));
}

template<typename Dtype>
void GPUDevice<Dtype>::sign(const int N, const Dtype* x, Dtype* y) {
  caffe_gpu_sign<Dtype>(N, x, y);
}

template<typename Dtype>
void GPUDevice<Dtype>::sgnbit(const int N, const Dtype* x, Dtype* y) {
  caffe_gpu_sgnbit<Dtype>(N, x, y);
}

template<typename Dtype>
void GPUDevice<Dtype>::fabs(const int N, const Dtype* x, Dtype* y) {
  caffe_gpu_fabs<Dtype>(N, x, y);
}

template<typename Dtype>
void GPUDevice<Dtype>::scale(const int N, const Dtype alpha, const Dtype *x,
                             Dtype* y) {
  this->copy(N, x, y);
  this->scal(N, alpha, y);
}

template<typename Dtype>
void GPUDevice<Dtype>::im2col(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  im2col_gpu(data_im, channels, height, width, ksize, pad, stride,
             data_col);
}

template<typename Dtype>
void GPUDevice<Dtype>::col2im(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im) {
  col2im_gpu(data_col, channels, height, width, psize, pad, stride,
             data_im);
}

INSTANTIATE_CLASS(GPUDevice);

}  // namespace caffe
