#include "caffe/backend/device.hpp"
#include "caffe/util/type_utils.hpp"
#include "caffe/quantizer.hpp"

#ifdef USE_LIBDNN  // USE_LIBDNN
#include "caffe/libdnn/libdnn_blas.hpp"
#endif  // USE_LIBDNN

namespace caffe {

template<typename Dtype>
void Device::copy(const uint_tp n, vptr<const Dtype> x, vptr<Dtype> y) {
  this->memcpy(safe_sizeof<Dtype>() * n, x, y);
}

template
void Device::copy(const uint_tp n, vptr<const half_fp> x,
                  vptr<half_fp> y);
template
void Device::copy(const uint_tp n, vptr<const float> x, vptr<float> y);
template
void Device::copy(const uint_tp n, vptr<const double> x, vptr<double> y);
template
void Device::copy(const uint_tp n, vptr<const uint8_t> x, vptr<uint8_t> y);
template
void Device::copy(const uint_tp n, vptr<const uint16_t> x, vptr<uint16_t> y);
template
void Device::copy(const uint_tp n, vptr<const uint32_t> x, vptr<uint32_t> y);
template
void Device::copy(const uint_tp n, vptr<const uint64_t> x, vptr<uint64_t> y);
template
void Device::copy(const uint_tp n, vptr<const int8_t> x, vptr<int8_t> y);
template
void Device::copy(const uint_tp n, vptr<const int16_t> x, vptr<int16_t> y);
template
void Device::copy(const uint_tp n, vptr<const int32_t> x, vptr<int32_t> y);
template
void Device::copy(const uint_tp n, vptr<const int64_t> x, vptr<int64_t> y);
template
void Device::copy(const uint_tp n, vptr<const void> x, vptr<void> y);


template<typename Dtype>
void Device::copy(const uint_tp n, const Dtype* x, vptr<Dtype> y) {
  this->memcpy(safe_sizeof<Dtype>() * n, x, y);
}

template
void Device::copy(const uint_tp n, const char* x, vptr<char> y);
template
void Device::copy(const uint_tp n, const half_fp* x,
                  vptr<half_fp> y);
template
void Device::copy(const uint_tp n, const float* x, vptr<float> y);
template
void Device::copy(const uint_tp n, const double* x, vptr<double> y);
template
void Device::copy(const uint_tp n, const uint8_t* x, vptr<uint8_t> y);
template
void Device::copy(const uint_tp n, const uint16_t* x, vptr<uint16_t> y);
template
void Device::copy(const uint_tp n, const uint32_t* x, vptr<uint32_t> y);
template
void Device::copy(const uint_tp n, const uint64_t* x, vptr<uint64_t> y);
template
void Device::copy(const uint_tp n, const int8_t* x, vptr<int8_t> y);
template
void Device::copy(const uint_tp n, const int16_t* x, vptr<int16_t> y);
template
void Device::copy(const uint_tp n, const int32_t* x, vptr<int32_t> y);
template
void Device::copy(const uint_tp n, const int64_t* x, vptr<int64_t> y);
template
void Device::copy(const uint_tp n, const void* x, vptr<void> y);


template<typename Dtype>
void Device::copy(const uint_tp n, vptr<const Dtype> x, Dtype* y) {
  this->memcpy(safe_sizeof<Dtype>() * n, x, y);
}

template
void Device::copy(const uint_tp n, vptr<const half_fp> x,
                  half_fp* y);
template
void Device::copy(const uint_tp n, vptr<const float> x, float* y);
template
void Device::copy(const uint_tp n, vptr<const double> x, double* y);
template
void Device::copy(const uint_tp n, vptr<const uint8_t> x, uint8_t* y);
template
void Device::copy(const uint_tp n, vptr<const uint16_t> x, uint16_t* y);
template
void Device::copy(const uint_tp n, vptr<const uint32_t> x, uint32_t* y);
template
void Device::copy(const uint_tp n, vptr<const uint64_t> x, uint64_t* y);
template
void Device::copy(const uint_tp n, vptr<const int8_t> x, int8_t* y);
template
void Device::copy(const uint_tp n, vptr<const int16_t> x, int16_t* y);
template
void Device::copy(const uint_tp n, vptr<const int32_t> x, int32_t* y);
template
void Device::copy(const uint_tp n, vptr<const int64_t> x, int64_t* y);
template
void Device::copy(const uint_tp n, vptr<const void> x, void* y);

#ifdef USE_HALF
template<>
void Device::gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                  const uint_tp m, const uint_tp n, const uint_tp k,
                  const half_fp alpha, vptr<const half_fp> a,
                  vptr<const half_fp> b,
                  const half_fp beta, vptr<half_fp> c,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const b_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const c_quant) {
  this->gemm_half(trans_a, trans_b, m, n, k, alpha, a, b, beta, c,
                  alpha_quant, a_quant, b_quant, beta_quant, c_quant);
}
template<>
void Device::gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                  const uint_tp n, const half_fp alpha,
                  vptr<const half_fp> a,
                  vptr<const half_fp> x, const half_fp beta,
                  vptr<half_fp> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const y_quant) {
  this->gemv_half(trans_a, m, n, alpha, a, x, beta, y,
                  alpha_quant, a_quant, x_quant, beta_quant, y_quant);
}
template<>
void Device::axpy(const uint_tp n, const half_fp alpha,
                  vptr<const half_fp> x, vptr<half_fp> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const y_quant) {
  this->axpy_half(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
template<>
void Device::axpby(const uint_tp n, const half_fp alpha,
                   vptr<const half_fp> x,
                   const half_fp beta, vptr<half_fp> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
  this->axpby_half(n, alpha, x, beta, y, alpha_quant, x_quant, beta_quant,
                   y_quant);
}
template<>
void Device::dot(const uint_tp n, vptr<const half_fp> x,
                    vptr<const half_fp> y, half_fp *out) {
  this->dot_half(n, x, y, out);
}
template<>
void Device::asum(const uint_tp n, vptr<const half_fp> x,
                       half_fp* y) {
  this->asum_half(n, x, y);
}
template<>
void Device::scal(const uint_tp n, const half_fp alpha,
                       vptr<half_fp> x) {
  this->scal_half(n, alpha, x);
}
template<>
void Device::scale(const uint_tp n, const half_fp alpha,
                   vptr<const half_fp> x, vptr<half_fp> y) {
  this->scale_half(n, alpha, x, y);
}
#endif  // USE_HALF

#ifdef USE_SINGLE
template<>
void Device::gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                  const uint_tp m, const uint_tp n, const uint_tp k,
                  const float alpha, vptr<const float> a,
                  vptr<const float> b, const float beta, vptr<float> c,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const b_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const c_quant) {
  this->gemm_float(trans_a, trans_b, m, n, k, alpha, a, b, beta, c,
                   alpha_quant, a_quant, b_quant, beta_quant, c_quant);
}
template<>
void Device::gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                  const uint_tp n, const float alpha,
                  vptr<const float> a,
                  vptr<const float> x, const float beta,
                  vptr<float> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const y_quant) {
  this->gemv_float(trans_a, m, n, alpha, a, x, beta, y,
                   alpha_quant, a_quant, x_quant, beta_quant, y_quant);
}
template<>
void Device::axpy(const uint_tp n, const float alpha,
                  vptr<const float> x, vptr<float> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const y_quant) {
  this->axpy_float(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
template<>
void Device::axpby(const uint_tp n, const float alpha, vptr<const float> x,
                   const float beta, vptr<float> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
  this->axpby_float(n, alpha, x, beta, y, alpha_quant, x_quant, beta_quant,
                    y_quant);
}
template<>
void Device::dot(const uint_tp n, vptr<const float> x, vptr<const float> y,
                 float *out) {
  this->dot_float(n, x, y, out);
}
template<>
void Device::asum(const uint_tp n, vptr<const float> x, float* y) {
  this->asum_float(n, x, y);
}
template<>
void Device::scale(const uint_tp n, const float alpha, vptr<const float> x,
                   vptr<float> y) {
  this->scale_float(n, alpha, x, y);
}
template<>
void Device::scal(const uint_tp n, const float alpha, vptr<float> x) {
  this->scal_float(n, alpha, x);
}
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
template<>
void Device::gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                  const uint_tp m, const uint_tp n, const uint_tp k,
                  const double alpha, vptr<const double> a,
                  vptr<const double> b, const double beta, vptr<double> c,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const b_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const c_quant) {
  this->gemm_double(trans_a, trans_b, m, n, k, alpha, a, b, beta, c,
                    alpha_quant, a_quant, b_quant, beta_quant, c_quant);
}
template<>
void Device::gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                  const uint_tp n, const double alpha,
                  vptr<const double> a,
                  vptr<const double> x, const double beta,
                  vptr<double> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const y_quant) {
  this->gemv_double(trans_a, m, n, alpha, a, x, beta, y,
                    alpha_quant, a_quant, x_quant, beta_quant, y_quant);
}
template<>
void Device::axpy(const uint_tp n, const double alpha,
                  vptr<const double> x, vptr<double> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const y_quant) {
  this->axpy_double(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
template<>
void Device::axpby(const uint_tp n, const double alpha,
                   vptr<const double> x,
                   const double beta, vptr<double> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
  this->axpby_double(n, alpha, x, beta, y, alpha_quant, x_quant,
                     beta_quant, y_quant);
}
template<>
void Device::dot(const uint_tp n, vptr<const double> x, vptr<const double> y,
                 double *out) {
  this->dot_double(n, x, y, out);
}
template<>
void Device::asum(const uint_tp n, vptr<const double> x, double* y) {
  this->asum_double(n, x, y);
}
template<>
void Device::scale(const uint_tp n, const double alpha,
                   vptr<const double> x, vptr<double> y) {
  this->scale_double(n, alpha, x, y);
}
template<>
void Device::scal(const uint_tp n, const double alpha, vptr<double> x) {
  this->scal_double(n, alpha, x);
}
#endif  // USE_DOUBLE

#ifdef USE_INT_QUANT_8
template<>
void Device::gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                  const uint_tp m, const uint_tp n, const uint_tp k,
                  const uint8_t alpha, vptr<const uint8_t> a,
                  vptr<const uint8_t> b, const uint8_t beta, vptr<uint8_t> c,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const b_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const c_quant) {
  this->gemm_uint8(trans_a, trans_b, m, n, k, alpha, a, b, beta, c,
                   alpha_quant, a_quant, b_quant, beta_quant, c_quant);
}
template<>
void Device::gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                  const uint_tp n, const uint8_t alpha,
                  vptr<const uint8_t> a,
                  vptr<const uint8_t> x, const uint8_t beta,
                  vptr<uint8_t> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const y_quant) {
  this->gemv_uint8(trans_a, m, n, alpha, a, x, beta, y,
                   alpha_quant, a_quant, x_quant, beta_quant, y_quant);
}
template<>
void Device::axpy(const uint_tp n, const uint8_t alpha,
                  vptr<const uint8_t> x, vptr<uint8_t> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const y_quant) {
  this->axpy_uint8(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
template<>
void Device::axpby(const uint_tp n, const uint8_t alpha,
                   vptr<const uint8_t> x,
                   const uint8_t beta, vptr<uint8_t> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
  this->axpby_uint8(n, alpha, x, beta, y, alpha_quant, x_quant, beta_quant,
                    y_quant);
}
template<>
void Device::dot(const uint_tp n, vptr<const uint8_t> x, vptr<const uint8_t> y,
                 uint8_t *out) {
  this->dot_uint8(n, x, y, out);
}
template<>
void Device::asum(const uint_tp n, vptr<const uint8_t> x, uint8_t* y) {
  this->asum_uint8(n, x, y);
}
template<>
void Device::scale(const uint_tp n, const uint8_t alpha,
                   vptr<const uint8_t> x, vptr<uint8_t> y) {
  this->scale_uint8(n, alpha, x, y);
}
template<>
void Device::scal(const uint_tp n, const uint8_t alpha, vptr<uint8_t> x) {
  this->scal_uint8(n, alpha, x);
}
#endif  // USE_INT_QUANT_8

#ifdef USE_INT_QUANT_16
template<>
void Device::gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                  const uint_tp m, const uint_tp n, const uint_tp k,
                  const uint16_t alpha, vptr<const uint16_t> a,
                  vptr<const uint16_t> b, const uint16_t beta,
                  vptr<uint16_t> c,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const b_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const c_quant) {
  this->gemm_uint16(trans_a, trans_b, m, n, k, alpha, a, b, beta, c,
                    alpha_quant, a_quant, b_quant, beta_quant, c_quant);
}
template<>
void Device::gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                  const uint_tp n, const uint16_t alpha,
                  vptr<const uint16_t> a,
                  vptr<const uint16_t> x, const uint16_t beta,
                  vptr<uint16_t> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const y_quant) {
  this->gemv_uint16(trans_a, m, n, alpha, a, x, beta, y,
                    alpha_quant, a_quant, x_quant, beta_quant, y_quant);
}
template<>
void Device::axpy(const uint_tp n, const uint16_t alpha,
                  vptr<const uint16_t> x, vptr<uint16_t> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const y_quant) {
  this->axpy_uint16(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
template<>
void Device::axpby(const uint_tp n, const uint16_t alpha,
                   vptr<const uint16_t> x,
                   const uint16_t beta, vptr<uint16_t> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
  this->axpby_uint16(n, alpha, x, beta, y, alpha_quant, x_quant, beta_quant,
                     y_quant);
}
template<>
void Device::dot(const uint_tp n, vptr<const uint16_t> x,
                 vptr<const uint16_t> y, uint16_t *out) {
  this->dot_uint16(n, x, y, out);
}
template<>
void Device::asum(const uint_tp n, vptr<const uint16_t> x, uint16_t* y) {
  this->asum_uint16(n, x, y);
}
template<>
void Device::scale(const uint_tp n, const uint16_t alpha,
                   vptr<const uint16_t> x, vptr<uint16_t> y) {
  this->scale_uint16(n, alpha, x, y);
}
template<>
void Device::scal(const uint_tp n, const uint16_t alpha, vptr<uint16_t> x) {
  this->scal_uint16(n, alpha, x);
}
#endif  // USE_INT_QUANT_16

#ifdef USE_INT_QUANT_32
template<>
void Device::gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                  const uint_tp m, const uint_tp n, const uint_tp k,
                  const uint32_t alpha, vptr<const uint32_t> a,
                  vptr<const uint32_t> b, const uint32_t beta,
                  vptr<uint32_t> c,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const b_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const c_quant) {
  this->gemm_uint32(trans_a, trans_b, m, n, k, alpha, a, b, beta, c,
                    alpha_quant, a_quant, b_quant, beta_quant, c_quant);
}
template<>
void Device::gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                  const uint_tp n, const uint32_t alpha,
                  vptr<const uint32_t> a,
                  vptr<const uint32_t> x, const uint32_t beta,
                  vptr<uint32_t> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const y_quant) {
  this->gemv_uint32(trans_a, m, n, alpha, a, x, beta, y,
                    alpha_quant, a_quant, x_quant, beta_quant, y_quant);
}
template<>
void Device::axpy(const uint_tp n, const uint32_t alpha,
                  vptr<const uint32_t> x, vptr<uint32_t> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const y_quant) {
  this->axpy_uint32(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
template<>
void Device::axpby(const uint_tp n, const uint32_t alpha,
                   vptr<const uint32_t> x,
                   const uint32_t beta, vptr<uint32_t> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
  this->axpby_uint32(n, alpha, x, beta, y, alpha_quant, x_quant, beta_quant,
                     y_quant);
}
template<>
void Device::dot(const uint_tp n, vptr<const uint32_t> x, vptr<const uint32_t> y,
                 uint32_t *out) {
  this->dot_uint32(n, x, y, out);
}
template<>
void Device::asum(const uint_tp n, vptr<const uint32_t> x, uint32_t* y) {
  this->asum_uint32(n, x, y);
}
template<>
void Device::scale(const uint_tp n, const uint32_t alpha,
                   vptr<const uint32_t> x, vptr<uint32_t> y) {
  this->scale_uint32(n, alpha, x, y);
}
template<>
void Device::scal(const uint_tp n, const uint32_t alpha, vptr<uint32_t> x) {
  this->scal_uint32(n, alpha, x);
}
#endif  // USE_INT_QUANT_32

#ifdef USE_INT_QUANT_64
template<>
void Device::gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                  const uint_tp m, const uint_tp n, const uint_tp k,
                  const uint64_t alpha, vptr<const uint64_t> a,
                  vptr<const uint64_t> b, const uint64_t beta,
                  vptr<uint64_t> c,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const b_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const c_quant) {
  this->gemm_uint64(trans_a, trans_b, m, n, k, alpha, a, b, beta, c,
                    alpha_quant, a_quant, b_quant, beta_quant, c_quant);
}
template<>
void Device::gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                  const uint_tp n, const uint64_t alpha,
                  vptr<const uint64_t> a,
                  vptr<const uint64_t> x, const uint64_t beta,
                  vptr<uint64_t> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const y_quant) {
  this->gemv_uint64(trans_a, m, n, alpha, a, x, beta, y,
                    alpha_quant, a_quant, x_quant, beta_quant, y_quant);
}
template<>
void Device::axpy(const uint_tp n, const uint64_t alpha,
                  vptr<const uint64_t> x, vptr<uint64_t> y,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const x_quant,
                  const QuantizerValues* const y_quant) {
  this->axpy_uint64(n, alpha, x, y, alpha_quant, x_quant, y_quant);
}
template<>
void Device::axpby(const uint_tp n, const uint64_t alpha,
                   vptr<const uint64_t> x,
                   const uint64_t beta, vptr<uint64_t> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
  this->axpby_uint64(n, alpha, x, beta, y, alpha_quant, x_quant,
                     beta_quant, y_quant);
}
template<>
void Device::dot(const uint_tp n, vptr<const uint64_t> x,
                 vptr<const uint64_t> y, uint64_t *out) {
  this->dot_uint64(n, x, y, out);
}
template<>
void Device::asum(const uint_tp n, vptr<const uint64_t> x, uint64_t* y) {
  this->asum_uint64(n, x, y);
}
template<>
void Device::scale(const uint_tp n, const uint64_t alpha,
                   vptr<const uint64_t> x, vptr<uint64_t> y) {
  this->scale_uint64(n, alpha, x, y);
}
template<>
void Device::scal(const uint_tp n, const uint64_t alpha, vptr<uint64_t> x) {
  this->scal_uint64(n, alpha, x);
}
#endif  // USE_INT_QUANT_64


}  // namespace caffe
