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
                          vptr<half_fp> y) {
  this->axpby_half(n, alpha, x, half_fp(1), y);
}
void Device::axpby_half(const uint_tp n, const half_fp alpha,
                   vptr<const half_fp> x,
                   const half_fp beta, vptr<half_fp> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->axpby(n, alpha, x,
                      beta, y, make_shared<Quantizer<half_fp, half_fp> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_half(const uint_tp n, vptr<const half_fp> x,
                         vptr<const half_fp> y,
                         half_fp* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->dot(n, x, y, out,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<half_fp, half_fp> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_half(const uint_tp n, vptr<const half_fp> x,
                           half_fp* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->asum(n, x, y,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<half_fp, half_fp> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_half(const uint_tp n, const half_fp alpha,
                           vptr<const half_fp> x,
                           vptr<half_fp> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->scale(n, alpha, x, y,
                               make_shared<Quantizer<half_fp, half_fp> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_half(const uint_tp n, const half_fp alpha, vptr<half_fp> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->scal(n, alpha, x,
                               make_shared<Quantizer<half_fp, half_fp> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_HALF


#ifdef USE_SINGLE
void Device::axpy_float(const uint_tp n, const float alpha,
                          vptr<const float> x,
                          vptr<float> y) {
  this->axpby_float(n, alpha, x, float(1), y);
}
void Device::axpby_float(const uint_tp n, const float alpha,
                   vptr<const float> x,
                   const float beta, vptr<float> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->axpby(n, alpha, x,
                      beta, y, make_shared<Quantizer<float, float> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_float(const uint_tp n, vptr<const float> x,
                         vptr<const float> y,
                         float* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->dot(n, x, y, out,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<float, float> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_float(const uint_tp n, vptr<const float> x,
                           float* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->asum(n, x, y,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<float, float> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_float(const uint_tp n, const float alpha,
                           vptr<const float> x,
                           vptr<float> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->scale(n, alpha, x, y,
                               make_shared<Quantizer<float, float> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_float(const uint_tp n, const float alpha, vptr<float> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->scal(n, alpha, x,
                               make_shared<Quantizer<float, float> >(this));
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
  this->template GetLibDNNBlas<double, double>()->axpby(n, alpha, x,
                      beta, y, make_shared<Quantizer<double, double> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_double(const uint_tp n, vptr<const double> x,
                         vptr<const double> y,
                         double* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->dot(n, x, y, out,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<double, double> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_double(const uint_tp n, vptr<const double> x,
                           double* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->asum(n, x, y,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<double, double> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_double(const uint_tp n, const double alpha,
                           vptr<const double> x,
                           vptr<double> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->scale(n, alpha, x, y,
                               make_shared<Quantizer<double, double> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_double(const uint_tp n, const double alpha, vptr<double> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->scal(n, alpha, x,
                               make_shared<Quantizer<double, double> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_DOUBLE

#ifdef USE_INT_QUANT_8
void Device::axpy_int8(const uint_tp n, const int8_t alpha,
                          vptr<const int8_t> x,
                          vptr<int8_t> y) {
  this->axpby_int8(n, alpha, x, int8_t(1), y);
}
void Device::axpby_int8(const uint_tp n, const int8_t alpha,
                   vptr<const int8_t> x,
                   const int8_t beta, vptr<int8_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int8_t, int8_t>()->axpby(n, alpha, x,
                      beta, y, make_shared<Quantizer<int8_t, int8_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_int8(const uint_tp n, vptr<const int8_t> x,
                         vptr<const int8_t> y,
                         int8_t* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int8_t, int8_t>()->dot(n, x, y, out,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int8_t, int8_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_int8(const uint_tp n, vptr<const int8_t> x,
                           int8_t* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int8_t, int8_t>()->asum(n, x, y,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int8_t, int8_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_int8(const uint_tp n, const int8_t alpha,
                           vptr<const int8_t> x,
                           vptr<int8_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int8_t, int8_t>()->scale(n, alpha, x, y,
                               make_shared<Quantizer<int8_t, int8_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_int8(const uint_tp n, const int8_t alpha, vptr<int8_t> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int8_t, int8_t>()->scal(n, alpha, x,
                               make_shared<Quantizer<int8_t, int8_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_8

#ifdef USE_INT_QUANT_16
void Device::axpy_int16(const uint_tp n, const int16_t alpha,
                          vptr<const int16_t> x,
                          vptr<int16_t> y) {
  this->axpby_int16(n, alpha, x, int16_t(1), y);
}
void Device::axpby_int16(const uint_tp n, const int16_t alpha,
                   vptr<const int16_t> x,
                   const int16_t beta, vptr<int16_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int16_t, int16_t>()->axpby(n, alpha, x,
                      beta, y, make_shared<Quantizer<int16_t, int16_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_int16(const uint_tp n, vptr<const int16_t> x,
                         vptr<const int16_t> y,
                         int16_t* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int16_t, int16_t>()->dot(n, x, y, out,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int16_t, int16_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_int16(const uint_tp n, vptr<const int16_t> x,
                           int16_t* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int16_t, int16_t>()->asum(n, x, y,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int16_t, int16_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_int16(const uint_tp n, const int16_t alpha,
                           vptr<const int16_t> x,
                           vptr<int16_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int16_t, int16_t>()->scale(n, alpha, x, y,
                               make_shared<Quantizer<int16_t, int16_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_int16(const uint_tp n, const int16_t alpha, vptr<int16_t> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int16_t, int16_t>()->scal(n, alpha, x,
                               make_shared<Quantizer<int16_t, int16_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_16

#ifdef USE_INT_QUANT_32
void Device::axpy_int32(const uint_tp n, const int32_t alpha,
                          vptr<const int32_t> x,
                          vptr<int32_t> y) {
  this->axpby_int32(n, alpha, x, int32_t(1), y);
}
void Device::axpby_int32(const uint_tp n, const int32_t alpha,
                   vptr<const int32_t> x,
                   const int32_t beta, vptr<int32_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int32_t, int32_t>()->axpby(n, alpha, x,
                      beta, y, make_shared<Quantizer<int32_t, int32_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_int32(const uint_tp n, vptr<const int32_t> x,
                         vptr<const int32_t> y,
                         int32_t* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int32_t, int32_t>()->dot(n, x, y, out,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int32_t, int32_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_int32(const uint_tp n, vptr<const int32_t> x,
                           int32_t* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int32_t, int32_t>()->asum(n, x, y,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int32_t, int32_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_int32(const uint_tp n, const int32_t alpha,
                           vptr<const int32_t> x,
                           vptr<int32_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int32_t, int32_t>()->scale(n, alpha, x, y,
                               make_shared<Quantizer<int32_t, int32_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_int32(const uint_tp n, const int32_t alpha, vptr<int32_t> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int32_t, int32_t>()->scal(n, alpha, x,
                               make_shared<Quantizer<int32_t, int32_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_32

#ifdef USE_INT_QUANT_64
void Device::axpy_int64(const uint_tp n, const int64_t alpha,
                          vptr<const int64_t> x,
                          vptr<int64_t> y) {
  this->axpby_int64(n, alpha, x, int64_t(1), y);
}
void Device::axpby_int64(const uint_tp n, const int64_t alpha,
                   vptr<const int64_t> x,
                   const int64_t beta, vptr<int64_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int64_t, int64_t>()->axpby(n, alpha, x,
                      beta, y, make_shared<Quantizer<int64_t, int64_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::dot_int64(const uint_tp n, vptr<const int64_t> x,
                         vptr<const int64_t> y,
                         int64_t* out) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int64_t, int64_t>()->dot(n, x, y, out,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int64_t, int64_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::asum_int64(const uint_tp n, vptr<const int64_t> x,
                           int64_t* y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int64_t, int64_t>()->asum(n, x, y,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int64_t, int64_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scale_int64(const uint_tp n, const int64_t alpha,
                           vptr<const int64_t> x,
                           vptr<int64_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int64_t, int64_t>()->scale(n, alpha, x, y,
                               make_shared<Quantizer<int64_t, int64_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
void Device::scal_int64(const uint_tp n, const int64_t alpha, vptr<int64_t> x) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int64_t, int64_t>()->scal(n, alpha, x,
                               make_shared<Quantizer<int64_t, int64_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_64


}  // namespace caffe
