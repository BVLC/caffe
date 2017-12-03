#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device.hpp"

namespace caffe {

void Device::axpy_half(const uint_tp n, const half_fp alpha,
                          vptr<const half_fp> x,
                          vptr<half_fp> y) {
  NOT_IMPLEMENTED;
}

void Device::axpy_float(const uint_tp n, const float alpha,
                           vptr<const float> x, vptr<float> y) {
  NOT_IMPLEMENTED;
}

void Device::axpy_double(const uint_tp n, const double alpha,
                            vptr<const double> x, vptr<double> y) {
  NOT_IMPLEMENTED;
}

void Device::axpby_half(const uint_tp n, const half_fp alpha,
                   vptr<const half_fp> x,
                   const half_fp beta, vptr<half_fp> y) {
  NOT_IMPLEMENTED;
}

void Device::axpby_float(const uint_tp n, const float alpha,
                   vptr<const float> x, const float beta, vptr<float> y) {
  NOT_IMPLEMENTED;
}

void Device::axpby_double(const uint_tp n, const double alpha,
                   vptr<const double> x, const double beta, vptr<double> y) {
  NOT_IMPLEMENTED;
}


void Device::scal_half(const uint_tp n, const half_fp alpha,
                  vptr<half_fp> x) {
  NOT_IMPLEMENTED;
}

void Device::scal_float(const uint_tp n, const float alpha, vptr<float> x) {
  NOT_IMPLEMENTED;
}

void Device::scal_double(const uint_tp n, const double alpha,
                             vptr<double> x) {
  NOT_IMPLEMENTED;
}

void Device::scal_int8(const uint_tp n, const int8_t alpha,
                             vptr<int8_t> x) {
  NOT_IMPLEMENTED;
}

void Device::scal_int16(const uint_tp n, const int16_t alpha,
                             vptr<int16_t> x) {
  NOT_IMPLEMENTED;
}

void Device::scal_int32(const uint_tp n, const int32_t alpha,
                             vptr<int32_t> x) {
  NOT_IMPLEMENTED;
}

void Device::scal_int64(const uint_tp n, const int64_t alpha,
                             vptr<int64_t> x) {
  NOT_IMPLEMENTED;
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
