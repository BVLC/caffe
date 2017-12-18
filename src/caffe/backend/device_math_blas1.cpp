#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device.hpp"

#ifdef USE_LIBDNN  // USE_LIBDNN
#include "caffe/libdnn/libdnn_blas.hpp"
#endif  // USE_LIBDNN

namespace caffe {

void Device::axpy_half(const uint_tp n, const half_fp alpha,
                          vptr<const half_fp> x,
                          vptr<half_fp> y) {
  this->axpby_half(n, alpha, x, half_fp(1), y);
}

void Device::axpy_float(const uint_tp n, const float alpha,
                           vptr<const float> x, vptr<float> y) {
  this->axpby_float(n, alpha, x, float(1), y);
}

void Device::axpy_double(const uint_tp n, const double alpha,
                            vptr<const double> x, vptr<double> y) {
  this->axpby_double(n, alpha, x, double(1), y);
}

void Device::axpby_half(const uint_tp n, const half_fp alpha,
                   vptr<const half_fp> x,
                   const half_fp beta, vptr<half_fp> y) {
#if defined(USE_LIBDNN) && defined(USE_HALF)
  this->template GetLibDNNBlas<half_fp, half_fp>()->axpby(n, alpha, x,
                      beta, y, make_shared<Quantizer<half_fp, half_fp> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}

void Device::axpby_float(const uint_tp n, const float alpha,
                   vptr<const float> x, const float beta, vptr<float> y) {
#if defined(USE_LIBDNN) && defined(USE_SINGLE)
  this->template GetLibDNNBlas<float, float>()->axpby(n, alpha, x,
                          beta, y, make_shared<Quantizer<float, float> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}

void Device::axpby_double(const uint_tp n, const double alpha,
                   vptr<const double> x, const double beta, vptr<double> y) {
#if defined(USE_LIBDNN) && defined(USE_DOUBLE)
  this->template GetLibDNNBlas<double, double>()->axpby(n, alpha, x,
                        beta, y, make_shared<Quantizer<double, double> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}


void Device::scal_half(const uint_tp n, const half_fp alpha, vptr<half_fp> x) {
#if defined(USE_LIBDNN) && defined(USE_HALF)
  this->template GetLibDNNBlas<half_fp, half_fp>()->scal(n, alpha, x,
                               make_shared<Quantizer<half_fp, half_fp> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}

void Device::scal_float(const uint_tp n, const float alpha, vptr<float> x) {
#if defined(USE_LIBDNN) && defined(USE_SINGLE)
  this->template GetLibDNNBlas<float, float>()->scal(n, alpha, x,
                                   make_shared<Quantizer<float, float> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}

void Device::scal_double(const uint_tp n, const double alpha,
                             vptr<double> x) {
#if defined(USE_LIBDNN) && defined(USE_DOUBLE)
  this->template GetLibDNNBlas<double, double>()->scal(n, alpha, x,
                                 make_shared<Quantizer<double, double> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}

void Device::scal_int8(const uint_tp n, const int8_t alpha,
                             vptr<int8_t> x) {
#if defined(USE_LIBDNN) && defined(USE_INT8_QUANT)
  this->template GetLibDNNBlas<int8_t, int8_t>()->scal(n, alpha, x,
                                 make_shared<Quantizer<int8_t, int8_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}

void Device::scal_int16(const uint_tp n, const int16_t alpha,
                             vptr<int16_t> x) {
#if defined(USE_LIBDNN) && defined(USE_INT16_QUANT)
  this->template GetLibDNNBlas<int16_t, int16_t>()->scal(n, alpha, x,
                               make_shared<Quantizer<int16_t, int16_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}

void Device::scal_int32(const uint_tp n, const int32_t alpha,
                             vptr<int32_t> x) {
#if defined(USE_LIBDNN) && defined(USE_INT32_QUANT)
  this->template GetLibDNNBlas<int32_t, int32_t>()->scal(n, alpha, x,
                               make_shared<Quantizer<int32_t, int32_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN;
}

void Device::scal_int64(const uint_tp n, const int64_t alpha,
                             vptr<int64_t> x) {
#if defined(USE_LIBDNN) && defined(USE_INT32_QUANT)
  this->template GetLibDNNBlas<int64_t, int64_t>()->scal(n, alpha, x,
                               make_shared<Quantizer<int64_t, int64_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN;
}

void Device::dot_half(const uint_tp n, vptr<const half_fp> x,
                         vptr<const half_fp> y,
                         half_fp* out) {
  NOT_IMPLEMENTED;
}
void Device::dot_float(const uint_tp n, vptr<const float> x,
                          vptr<const float> y, float* out) {
  NOT_IMPLEMENTED;
}
void Device::dot_double(const uint_tp n, vptr<const double> x,
                           vptr<const double> y, double* out) {
  NOT_IMPLEMENTED;
}

void Device::asum_half(const uint_tp n, vptr<const half_fp> x,
                           half_fp* y) {
  NOT_IMPLEMENTED;
}

void Device::asum_float(const uint_tp n, vptr<const float> x, float* y) {
  NOT_IMPLEMENTED;
}


void Device::asum_double(const uint_tp n, vptr<const double> x, double* y) {
  NOT_IMPLEMENTED;
}


void Device::scale_half(const uint_tp n, const half_fp alpha,
                           vptr<const half_fp> x,
                           vptr<half_fp> y) {
  NOT_IMPLEMENTED;
}

void Device::scale_float(const uint_tp n, const float alpha,
                             vptr<const float> x, vptr<float> y) {
  NOT_IMPLEMENTED;
}

void Device::scale_double(const uint_tp n, const double alpha,
                             vptr<const double> x, vptr<double> y) {
  NOT_IMPLEMENTED;
}

}  // namespace caffe
