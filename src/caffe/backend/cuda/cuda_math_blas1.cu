#ifdef USE_CUDA
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>

#include "caffe/backend/cuda/cuda_device.hpp"
#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/cuda/caffe_cuda.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"

#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

namespace caffe {

#ifdef USE_SINGLE
void CudaDevice::axpy_float(const uint_tp n, const float alpha,
                            vptr<const float> x, vptr<float> y,
                            const QuantizerValues* const alpha_quant,
                            const QuantizerValues* const x_quant,
                            const QuantizerValues* const y_quant) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), n, &alpha,
                           x.get_cuda_ptr(), 1, y.get_cuda_ptr(), 1));
}
void CudaDevice::axpby_float(const uint_tp n, const float alpha,
                             vptr<const float> x, const float beta,
                             vptr<float> y,
                             const QuantizerValues* const alpha_quant,
                             const QuantizerValues* const x_quant,
                             const QuantizerValues* const beta_quant,
                             const QuantizerValues* const y_quant) {
  this->scal_float(n, beta, y);
  this->axpy_float(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
void CudaDevice::scal_float(const uint_tp n, const float alpha,
                            vptr<float> x) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(),
                           n, &alpha, x.get_cuda_ptr(), 1));
}
void CudaDevice::dot_float(const uint_tp n, vptr<const float> x,
                           vptr<const float> y, float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x.get_cuda_ptr(), 1,
                          y.get_cuda_ptr(), 1, out));
}
void CudaDevice::asum_float(const uint_tp n, vptr<const float> x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x.get_cuda_ptr(),
                           1, y));
}
void CudaDevice::scale_float(const uint_tp n, const float alpha,
                             vptr<const float> x, vptr<float> y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x.get_cuda_ptr(), 1,
                           y.get_cuda_ptr(), 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y.get_cuda_ptr(),
                           1));
}
template <>
void CudaDevice::scal_str<float>(const int_tp n, const float alpha,
                                 vptr<float> x, cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, x.get_cuda_ptr(),
                           1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
void CudaDevice::axpy_double(const uint_tp n, const double alpha,
                             vptr<const double> x, vptr<double> y,
                             const QuantizerValues* const alpha_quant,
                             const QuantizerValues* const x_quant,
                             const QuantizerValues* const y_quant) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), n, &alpha,
                           x.get_cuda_ptr(), 1, y.get_cuda_ptr(), 1));
}
void CudaDevice::axpby_double(const uint_tp n, const double alpha,
                              vptr<const double> x,
                              const double beta, vptr<double> y,
                              const QuantizerValues* const alpha_quant,
                              const QuantizerValues* const x_quant,
                              const QuantizerValues* const beta_quant,
                              const QuantizerValues* const y_quant) {
  this->scal_double(n, beta, y);
  this->axpy_double(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
void CudaDevice::scal_double(const uint_tp n, const double alpha,
                             vptr<double> x) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha,
                           x.get_cuda_ptr(), 1));
}
void CudaDevice::dot_double(const uint_tp n, vptr<const double> x,
                            vptr<const double> y, double* out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x.get_cuda_ptr(), 1,
                          y.get_cuda_ptr(), 1, out));
}
void CudaDevice::asum_double(const uint_tp n, vptr<const double> x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x.get_cuda_ptr(),
                           1, y));
}
void CudaDevice::scale_double(const uint_tp n, const double alpha,
                              vptr<const double> x, vptr<double> y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x.get_cuda_ptr(), 1,
                           y.get_cuda_ptr(), 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y.get_cuda_ptr(),
                           1));
}
template <>
void CudaDevice::scal_str<double>(const int_tp n, const double alpha,
                                  vptr<double> x, cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, x.get_cuda_ptr(),
                           1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}
#endif  // USE_DOUBLE

}  // namespace caffe
#endif  // USE_CUDA
