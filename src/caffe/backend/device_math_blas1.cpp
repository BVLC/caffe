#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device.hpp"

#ifdef USE_LIBDNN  // USE_LIBDNN
#include "caffe/libdnn/libdnn_blas.hpp"
#endif  // USE_LIBDNN

namespace caffe {

#ifdef USE_HALF
void Device::axpy_half(const uint_tp n, const half_fp alpha,
                          vptr<const half_fp> x,
                          vptr<half_fp> y,
                          const QuantizerValues* const alpha_quant,
                          const QuantizerValues* const x_quant,
                          const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->axpby(n, alpha, x,
                                     half_fp(1), y,
                                     alpha_quant, x_quant, nullptr, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::axpby_half(const uint_tp n, const half_fp alpha,
                   vptr<const half_fp> x,
                   const half_fp beta, vptr<half_fp> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->axpby(n, alpha, x, beta, y,
                                     alpha_quant, x_quant, beta_quant, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_half(const uint_tp n, vptr<const half_fp> x,
                         vptr<const half_fp> y,
                         half_fp* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->dot(n, x, y, out);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_half(const uint_tp n, vptr<const half_fp> x,
                           half_fp* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->asum(n, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_half(const uint_tp n, const half_fp alpha,
                           vptr<const half_fp> x,
                           vptr<half_fp> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->scale(n, alpha, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_half(const uint_tp n, const half_fp alpha, vptr<half_fp> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->scal(n, alpha, x);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_HALF


#ifdef USE_SINGLE
void Device::axpy_float(const uint_tp n, const float alpha,
                        vptr<const float> x, vptr<float> y,
                        const QuantizerValues* const alpha_quant,
                        const QuantizerValues* const x_quant,
                        const QuantizerValues* const y_quant) {
  this->axpby_float(n, alpha, x, float(1), y, alpha_quant, x_quant,
                    nullptr, y_quant);
}
void Device::axpby_float(const uint_tp n, const float alpha,
                   vptr<const float> x, const float beta, vptr<float> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->axpby(n, alpha, x, beta, y,
                                     alpha_quant, x_quant, beta_quant, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_float(const uint_tp n, vptr<const float> x,
                         vptr<const float> y,
                         float* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->dot(n, x, y, out);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_float(const uint_tp n, vptr<const float> x, float* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->asum(n, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_float(const uint_tp n, const float alpha,
                           vptr<const float> x,
                           vptr<float> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->scale(n, alpha, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_float(const uint_tp n, const float alpha, vptr<float> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->scal(n, alpha, x);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
void Device::axpy_double(const uint_tp n, const double alpha,
                          vptr<const double> x,
                          vptr<double> y) {
  this->axpby_double(n, alpha, x, double(1), y);
}
void Device::axpby_double(const uint_tp n, const double alpha,
                   vptr<const double> x,
                   const double beta, vptr<double> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->axpby(n, alpha, x);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_double(const uint_tp n, vptr<const double> x,
                         vptr<const double> y,
                         double* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->dot(n, x, y, out);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_double(const uint_tp n, vptr<const double> x,
                           double* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->asum(n, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_double(const uint_tp n, const double alpha,
                           vptr<const double> x,
                           vptr<double> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->scale(n, alpha, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_double(const uint_tp n, const double alpha, vptr<double> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->scal(n, alpha, x);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_DOUBLE

#ifdef USE_INT_QUANT_8
void Device::axpy_uint8(const uint_tp n, const uint8_t alpha,
                        vptr<const uint8_t> x, vptr<uint8_t> y,
                        const QuantizerValues* const alpha_quant,
                        const QuantizerValues* const x_quant,
                        const QuantizerValues* const y_quant) {
  this->axpby_uint8(n, alpha, x, uint8_t(1), y, alpha_quant, x_quant,
                    nullptr, y_quant);
}
void Device::axpby_uint8(const uint_tp n, const uint8_t alpha,
                   vptr<const uint8_t> x, const uint8_t beta, vptr<uint8_t> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint8_t, uint8_t>()->axpby(n, alpha, x, beta, y,
                                     alpha_quant, x_quant, beta_quant, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_uint8(const uint_tp n, vptr<const uint8_t> x,
                         vptr<const uint8_t> y,
                         uint8_t* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint8_t, uint8_t>()->dot(n, x, y, out);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_uint8(const uint_tp n, vptr<const uint8_t> x,
                           uint8_t* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint8_t, uint8_t>()->asum(n, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_uint8(const uint_tp n, const uint8_t alpha,
                           vptr<const uint8_t> x,
                           vptr<uint8_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint8_t, uint8_t>()->scale(n, alpha, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_uint8(const uint_tp n, const uint8_t alpha, vptr<uint8_t> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint8_t, uint8_t>()->scal(n, alpha, x);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_8

#ifdef USE_INT_QUANT_16
void Device::axpy_uint16(const uint_tp n, const uint16_t alpha,
                         vptr<const uint16_t> x, vptr<uint16_t> y,
                         const QuantizerValues* const alpha_quant,
                         const QuantizerValues* const x_quant,
                         const QuantizerValues* const y_quant) {
  this->axpby_uint16(n, alpha, x, uint16_t(1), y,
                     alpha_quant, x_quant, nullptr, y_quant);
}
void Device::axpby_uint16(const uint_tp n, const uint16_t alpha,
                   vptr<const uint16_t> x, const uint16_t beta,
                   vptr<uint16_t> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint16_t, uint16_t>()->axpby(n, alpha, x,
                                           beta, y, alpha_quant, x_quant,
                                           beta_quant, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_uint16(const uint_tp n, vptr<const uint16_t> x,
                         vptr<const uint16_t> y,
                         uint16_t* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint16_t, uint16_t>()->dot(n, x, y, out);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_uint16(const uint_tp n, vptr<const uint16_t> x,
                           uint16_t* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint16_t, uint16_t>()->asum(n, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_uint16(const uint_tp n, const uint16_t alpha,
                           vptr<const uint16_t> x,
                           vptr<uint16_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint16_t, uint16_t>()->scale(n, alpha, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_uint16(const uint_tp n, const uint16_t alpha, vptr<uint16_t> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint16_t, uint16_t>()->scal(n, alpha, x);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_16

#ifdef USE_INT_QUANT_32
void Device::axpy_uint32(const uint_tp n, const uint32_t alpha,
                         vptr<const uint32_t> x,
                         vptr<uint32_t> y,
                         const QuantizerValues* const alpha_quant,
                         const QuantizerValues* const x_quant,
                         const QuantizerValues* const y_quant) {
  this->axpby_uint32(n, alpha, x, uint32_t(1), y,
                     alpha_quant, x_quant, nullptr, y_quant);
}
void Device::axpby_uint32(const uint_tp n, const uint32_t alpha,
                   vptr<const uint32_t> x,
                   const uint32_t beta, vptr<uint32_t> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint32_t, uint32_t>()->axpby(n, alpha, x,
                                      beta, y, alpha_quant, x_quant, beta_quant,
                                      y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_uint32(const uint_tp n, vptr<const uint32_t> x,
                         vptr<const uint32_t> y,
                         uint32_t* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint32_t, uint32_t>()->dot(n, x, y, out);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_uint32(const uint_tp n, vptr<const uint32_t> x,
                           uint32_t* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint32_t, uint32_t>()->asum(n, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_uint32(const uint_tp n, const uint32_t alpha,
                           vptr<const uint32_t> x,
                           vptr<uint32_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint32_t, uint32_t>()->scale(n, alpha, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_uint32(const uint_tp n, const uint32_t alpha,
                         vptr<uint32_t> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint32_t, uint32_t>()->scal(n, alpha, x);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_32

#ifdef USE_INT_QUANT_64
void Device::axpy_uint64(const uint_tp n, const uint64_t alpha,
                         vptr<const uint64_t> x, vptr<uint64_t> y,
                         const QuantizerValues* const alpha_quant,
                         const QuantizerValues* const x_quant,
                         const QuantizerValues* const y_quant) {
  this->axpby_uint64(n, alpha, x, uint64_t(1), y, alpha_quant, x_quant, nullptr,
                     y_quant);
}
void Device::axpby_uint64(const uint_tp n, const uint64_t alpha,
                   vptr<const uint64_t> x,  const uint64_t beta,
                   vptr<uint64_t> y,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const x_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint64_t, uint64_t>()->axpby(n, alpha, x,
                            beta, y, alpha_quant, x_quant, beta_quant, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_uint64(const uint_tp n, vptr<const uint64_t> x,
                         vptr<const uint64_t> y,
                         uint64_t* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint64_t, uint64_t>()->dot(n, x, y, out);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_uint64(const uint_tp n, vptr<const uint64_t> x,
                           uint64_t* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint64_t, uint64_t>()->asum(n, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_uint64(const uint_tp n, const uint64_t alpha,
                           vptr<const uint64_t> x,
                           vptr<uint64_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint64_t, uint64_t>()->scale(n, alpha, x, y);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_uint64(const uint_tp n,
                         const uint64_t alpha, vptr<uint64_t> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint64_t, uint64_t>()->scal(n, alpha, x);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_64


}  // namespace caffe
