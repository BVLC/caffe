#include "caffe/backend/device.hpp"
#include "caffe/util/type_utils.hpp"

namespace caffe {

#ifdef USE_HALF
template<>
void Device::rng_uniform(const uint_tp n, const half_fp a,
                      const half_fp b, vptr<half_fp> r) {
  this->rng_uniform_half(n, a, b, r);
}
void Device::rng_uniform_half(const uint_tp n, const half_fp a,
                              const half_fp b, vptr<half_fp> r) {
  vector<half_fp> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  this->memcpy(sizeof(half_fp) * n, &random[0], vptr<void>(r));
}
template<>
void Device::rng_gaussian(const uint_tp n, const half_fp mu,
                          const half_fp sigma, vptr<half_fp> r) {
  this->rng_gaussian_half(n, mu, sigma, r);
}
void Device::rng_gaussian_half(const uint_tp n, const half_fp mu,
                               const half_fp sigma, vptr<half_fp> r) {
  vector<half_fp> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  this->memcpy(sizeof(half_fp) * n, &random[0], vptr<void>(r));
}
template<>
void Device::rng_bernoulli(const uint_tp n, const half_fp p,
                           vptr<int8_t> r) {
  this->rng_bernoulli_half(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const half_fp p,
                           vptr<int16_t> r) {
  this->rng_bernoulli_half(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const half_fp p,
                           vptr<int32_t> r) {
  this->rng_bernoulli_half(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const half_fp p,
                           vptr<int64_t> r) {
  this->rng_bernoulli_half(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const half_fp p,
                           vptr<uint8_t> r) {
  this->rng_bernoulli_half(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const half_fp p,
                           vptr<uint16_t> r) {
  this->rng_bernoulli_half(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const half_fp p,
                           vptr<uint32_t> r) {
  this->rng_bernoulli_half(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const half_fp p,
                           vptr<uint64_t> r) {
  this->rng_bernoulli_half(n, p, r);
}
void Device::rng_bernoulli_half(const uint_tp n, const half_fp p,
                                    vptr<int8_t> r) {
  vector<int8_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int8_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_half(const uint_tp n, const half_fp p,
                                    vptr<int16_t> r) {
  vector<int16_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int16_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_half(const uint_tp n, const half_fp p,
                                    vptr<int32_t> r) {
  vector<int32_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int32_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_half(const uint_tp n, const half_fp p,
                                    vptr<int64_t> r) {
  vector<int64_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int64_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_half(const uint_tp n, const half_fp p,
                                    vptr<uint8_t> r) {
  vector<uint8_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint8_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_half(const uint_tp n, const half_fp p,
                                    vptr<uint16_t> r) {
  vector<uint16_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint16_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_half(const uint_tp n, const half_fp p,
                                    vptr<uint32_t> r) {
  vector<uint32_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint32_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_half(const uint_tp n, const half_fp p,
                                    vptr<uint64_t> r) {
  vector<uint64_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint64_t) * n, &random[0], vptr<void>(r));
}

#endif  // USE_HALF

#ifdef USE_SINGLE
template<>
void Device::rng_uniform(const uint_tp n, const float a, const float b,
                               vptr<float> r) {
  this->rng_uniform_float(n, a, b, r);
}
void Device::rng_uniform_float(const uint_tp n, const float a,
                                   const float b, vptr<float> r) {
  vector<float> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  this->memcpy(sizeof(float) * n, &random[0], vptr<void>(r));
}
template<>
void Device::rng_gaussian(const uint_tp n, const float mu,
                                const float sigma, vptr<float> r) {
  this->rng_gaussian_float(n, mu, sigma, r);
}
void Device::rng_gaussian_float(const uint_tp n, const float mu,
                                   const float sigma, vptr<float> r) {
  vector<float> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  this->memcpy(sizeof(float) * n, &random[0], vptr<void>(r));
}

template<>
void Device::rng_bernoulli(const uint_tp n, const float p,
                           vptr<int8_t> r) {
  this->rng_bernoulli_float(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const float p,
                           vptr<int16_t> r) {
  this->rng_bernoulli_float(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const float p,
                           vptr<int32_t> r) {
  this->rng_bernoulli_float(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const float p,
                           vptr<int64_t> r) {
  this->rng_bernoulli_float(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const float p,
                           vptr<uint8_t> r) {
  this->rng_bernoulli_float(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const float p,
                           vptr<uint16_t> r) {
  this->rng_bernoulli_float(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const float p,
                           vptr<uint32_t> r) {
  this->rng_bernoulli_float(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const float p,
                           vptr<uint64_t> r) {
  this->rng_bernoulli_float(n, p, r);
}
void Device::rng_bernoulli_float(const uint_tp n, const float p,
                                    vptr<int8_t> r) {
  vector<int8_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int8_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_float(const uint_tp n, const float p,
                                    vptr<int16_t> r) {
  vector<int16_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int16_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_float(const uint_tp n, const float p,
                                    vptr<int32_t> r) {
  vector<int32_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int32_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_float(const uint_tp n, const float p,
                                    vptr<int64_t> r) {
  vector<int64_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int64_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_float(const uint_tp n, const float p,
                                    vptr<uint8_t> r) {
  vector<uint8_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint8_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_float(const uint_tp n, const float p,
                                    vptr<uint16_t> r) {
  vector<uint16_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint16_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_float(const uint_tp n, const float p,
                                    vptr<uint32_t> r) {
  vector<uint32_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint32_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_float(const uint_tp n, const float p,
                                    vptr<uint64_t> r) {
  vector<uint64_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint64_t) * n, &random[0], vptr<void>(r));
}
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
template<>
void Device::rng_uniform(const uint_tp n, const double a,
                                const double b, vptr<double> r) {
  this->rng_uniform_double(n, a, b, r);
}
void Device::rng_uniform_double(const uint_tp n, const double a,
                                   const double b, vptr<double> r) {
  vector<double> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  this->memcpy(sizeof(double) * n, &random[0], vptr<void>(r));
}
template<>
void Device::rng_gaussian(const uint_tp n, const double mu,
                                 const double sigma, vptr<double> r) {
  this->rng_gaussian_double(n, mu, sigma, r);
}
void Device::rng_gaussian_double(const uint_tp n, const double mu,
                                     const double sigma, vptr<double> r) {
  vector<double> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  this->memcpy(sizeof(double) * n, &random[0], vptr<void>(r));
}
template<>
void Device::rng_bernoulli(const uint_tp n, const double p,
                           vptr<int8_t> r) {
  this->rng_bernoulli_double(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const double p,
                           vptr<int16_t> r) {
  this->rng_bernoulli_double(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const double p,
                           vptr<int32_t> r) {
  this->rng_bernoulli_double(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const double p,
                           vptr<int64_t> r) {
  this->rng_bernoulli_double(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const double p,
                           vptr<uint8_t> r) {
  this->rng_bernoulli_double(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const double p,
                           vptr<uint16_t> r) {
  this->rng_bernoulli_double(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const double p,
                           vptr<uint32_t> r) {
  this->rng_bernoulli_double(n, p, r);
}
template<>
void Device::rng_bernoulli(const uint_tp n, const double p,
                           vptr<uint64_t> r) {
  this->rng_bernoulli_double(n, p, r);
}
void Device::rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int8_t> r) {
  vector<int8_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int8_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int16_t> r) {
  vector<int16_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int16_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int32_t> r) {
  vector<int32_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int32_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int64_t> r) {
  vector<int64_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(int64_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<uint8_t> r) {
  vector<uint8_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint8_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<uint16_t> r) {
  vector<uint16_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint16_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<uint32_t> r) {
  vector<uint32_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint32_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<uint64_t> r) {
  vector<uint64_t> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(uint64_t) * n, &random[0], vptr<void>(r));
}
#endif  // USE_DOUBLE

#ifdef USE_INT_QUANT_8
template<>
void Device::rng_uniform(const uint_tp n, const uint8_t a,
                                const uint8_t b, vptr<uint8_t> r) {
  this->rng_uniform_uint8(n, a, b, r);
}
void Device::rng_uniform_uint8(const uint_tp n, const uint8_t a,
                               const uint8_t b, vptr<uint8_t> r) {
  vector<uint8_t> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  this->memcpy(sizeof(uint8_t) * n, &random[0], vptr<void>(r));
}
#endif  // USE_INT_QUANT_8

#ifdef USE_INT_QUANT_16
template<>
void Device::rng_uniform(const uint_tp n, const uint16_t a,
                                const uint16_t b, vptr<uint16_t> r) {
  this->rng_uniform_uint16(n, a, b, r);
}
void Device::rng_uniform_uint16(const uint_tp n, const uint16_t a,
                                const uint16_t b, vptr<uint16_t> r) {
  vector<uint16_t> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  this->memcpy(sizeof(uint16_t) * n, &random[0], vptr<void>(r));
}
#endif  // USE_INT_QUANT_16

#ifdef USE_INT_QUANT_32
template<>
void Device::rng_uniform(const uint_tp n, const uint32_t a,
                         const uint32_t b, vptr<uint32_t> r) {
  this->rng_uniform_uint32(n, a, b, r);
}
void Device::rng_uniform_uint32(const uint_tp n, const uint32_t a,
                                const uint32_t b, vptr<uint32_t> r) {
  vector<uint32_t> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  this->memcpy(sizeof(uint32_t) * n, &random[0], vptr<void>(r));
}
#endif  // USE_INT_QUANT_32


#ifdef USE_INT_QUANT_64
template<>
void Device::rng_uniform(const uint_tp n, const uint64_t a,
                         const uint64_t b, vptr<uint64_t> r) {
  this->rng_uniform_uint64(n, a, b, r);
}
void Device::rng_uniform_uint64(const uint_tp n, const uint64_t a,
                                const uint64_t b, vptr<uint64_t> r) {
  vector<uint64_t> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  this->memcpy(sizeof(uint64_t) * n, &random[0], vptr<void>(r));
}
#endif  // USE_INT_QUANT_64

void Device::rng_uniform(const uint_tp n, vptr<uint8_t> r) {
  vector<uint8_t> random(n);  //NOLINT
  caffe_rng_uniform(n, &random[0]);
  this->memcpy(sizeof(uint8_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_uniform(const uint_tp n, vptr<uint16_t> r) {
  vector<uint16_t> random(n);  //NOLINT
  caffe_rng_uniform(n, &random[0]);
  this->memcpy(sizeof(uint16_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_uniform(const uint_tp n, vptr<uint32_t> r) {
  vector<uint32_t> random(n);  //NOLINT
  caffe_rng_uniform(n, &random[0]);
  this->memcpy(sizeof(uint32_t) * n, &random[0], vptr<void>(r));
}
void Device::rng_uniform(const uint_tp n, vptr<uint64_t> r) {
  vector<uint64_t> random(n);  //NOLINT
  caffe_rng_uniform(n, &random[0]);
  this->memcpy(sizeof(uint64_t) * n, &random[0], vptr<void>(r));
}


}  // namespace caffe
