#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>

#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/cuda/caffe_cuda.hpp"
#include "caffe/backend/cuda/cuda_device.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"

#ifdef USE_CUDA

#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#endif  // USE_CUDA

namespace caffe {

#ifdef USE_CUDA

void cuda_device::axpy_half(const uint_tp N,
                       const half_float::half alpha,
                       vptr<half_float::half> X,
                       vptr<half_float::half> Y) {
#ifdef USE_GPU_HALF
  NOT_IMPLEMENTED;  // TODO
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif  // USE_GPU_HALF
}

void cuda_device::axpy_float(const uint_tp N, const float alpha,
                        vptr<float> X, vptr<float> Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha,
                           X.get_cuda_ptr(), 1, Y.get_cuda_ptr(), 1));
}

void cuda_device::axpy_double(const uint_tp N, const double alpha,
                        vptr<double> X, vptr<double> Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha,
                           X.get_cuda_ptr(), 1, Y.get_cuda_ptr(), 1));
}

void cuda_device::axpby_half(const uint_tp N, const half_float::half alpha,
                   vptr<half_float::half> X,
                   const half_float::half beta, vptr<half_float::half> Y) {
  cuda_device::scal_half(N, beta, Y);
  cuda_device::axpy_half(N, alpha, X, Y);
}

void cuda_device::axpby_float(const uint_tp N, const float alpha,
                   vptr<float> X, const float beta, vptr<float> Y) {
  cuda_device::scal_float(N, beta, Y);
  cuda_device::axpy_float(N, alpha, X, Y);
}

void cuda_device::axpby_double(const uint_tp N, const double alpha,
                   vptr<double> X, const double beta, vptr<double> Y) {
  cuda_device::scal_double(N, beta, Y);
  cuda_device::axpy_double(N, alpha, X, Y);
}

void cuda_device::scal_half(const uint_tp N, const half_float::half alpha,
                             vptr<half_float::half> X) {
#ifdef USE_GPU_HALF
  NOT_IMPLEMENTED;  // TODO
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif  // USE_GPU_HALF
}

void cuda_device::scal_float(const uint_tp N, const float alpha,
                             vptr<float> X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(),
                           N, &alpha, X.get_cuda_ptr(), 1));
}

void cuda_device::scal_double(const uint_tp N, const double alpha,
                               vptr<double> X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha,
                           X.get_cuda_ptr(), 1));
}

template <>
void cuda_device::scal_str<half_float::half>(const int_tp N,
                                             const half_float::half alpha,
                                             vptr<half_float::half> X,
                                             cudaStream_t str) {
#ifdef USE_GPU_HALF
  NOT_IMPLEMENTED;  // TODO
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif  // USE_GPU_HALF
}

template <>
void cuda_device::scal_str<float>(const int_tp N, const float alpha,
                                  vptr<float> X,
                                  cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X.get_cuda_ptr(),
                           1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void cuda_device::scal_str<double>(const int_tp N, const double alpha,
                                   vptr<double> X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X.get_cuda_ptr(),
                           1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

void cuda_device::dot_half(const uint_tp n, vptr<half_float::half> x,
                           vptr<half_float::half> y,
                           half_float::half* out) {
#ifdef USE_GPU_HALF
  NOT_IMPLEMENTED;  // TODO
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif  // USE_GPU_HALF
}

void cuda_device::dot_float(const uint_tp n, vptr<float> x, vptr<float> y,
                          float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x.get_cuda_ptr(), 1,
                          y.get_cuda_ptr(), 1, out));
}

void cuda_device::dot_double(const uint_tp n, vptr<double> x,
                             vptr<double> y, double* out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x.get_cuda_ptr(), 1,
                          y.get_cuda_ptr(), 1, out));
}

void cuda_device::asum_half(const uint_tp n, vptr<half_float::half> x,
                            half_float::half *y) {
#ifdef USE_GPU_HALF
  NOT_IMPLEMENTED;  // TODO
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif  // USE_GPU_HALF
}

void cuda_device::asum_float(const uint_tp n, vptr<float> x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x.get_cuda_ptr(),
                           1, y));
}

void cuda_device::asum_double(const uint_tp n, vptr<double> x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x.get_cuda_ptr(),
                           1, y));
}

void cuda_device::scale_half(const uint_tp n, const half_float::half alpha,
                           vptr<half_float::half> x, vptr<half_float::half> y) {
#ifdef USE_GPU_HALF
  NOT_IMPLEMENTED;  // TODO
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif  // USE_GPU_HALF
}

void cuda_device::scale_float(const uint_tp n, const float alpha, vptr<float> x,
                            vptr<float> y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x.get_cuda_ptr(), 1,
                           y.get_cuda_ptr(), 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y.get_cuda_ptr(),
                           1));
}

void cuda_device::scale_double(const uint_tp n, const double alpha,
                             vptr<double> x, vptr<double> y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x.get_cuda_ptr(), 1,
                           y.get_cuda_ptr(), 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y.get_cuda_ptr(),
                           1));
}


#endif  // USE_CUDA

}
