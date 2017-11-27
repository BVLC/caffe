#include "caffe/backend/device.hpp"
#include "caffe/util/type_utils.hpp"

namespace caffe {

template<typename Dtype>
void Device::copy(const uint_tp n, vptr<const Dtype> x, vptr<Dtype> y) {
  this->memcpy(safe_sizeof<Dtype>() * n, x, y);
}

template<>
void Device::copy(const uint_tp n, vptr<const half_fp> x,
                  vptr<half_fp> y);
template<>
void Device::copy(const uint_tp n, vptr<const float> x, vptr<float> y);
template<>
void Device::copy(const uint_tp n, vptr<const double> x, vptr<double> y);
template<>
void Device::copy(const uint_tp n, vptr<const int8_t> x, vptr<int8_t> y);
template<>
void Device::copy(const uint_tp n, vptr<const int16_t> x, vptr<int16_t> y);
template<>
void Device::copy(const uint_tp n, vptr<const int32_t> x, vptr<int32_t> y);
template<>
void Device::copy(const uint_tp n, vptr<const int64_t> x, vptr<int64_t> y);
template<>
void Device::copy(const uint_tp n, vptr<const uint8_t> x, vptr<uint8_t> y);
template<>
void Device::copy(const uint_tp n, vptr<const uint16_t> x, vptr<uint16_t> y);
template<>
void Device::copy(const uint_tp n, vptr<const uint32_t> x, vptr<uint32_t> y);
template<>
void Device::copy(const uint_tp n, vptr<const uint64_t> x, vptr<uint64_t> y);
template<>
void Device::copy(const uint_tp n, vptr<const void> x, vptr<void> y);


template<typename Dtype>
void Device::copy(const uint_tp n, const Dtype* x, vptr<Dtype> y) {
  this->memcpy(safe_sizeof<Dtype>() * n, x, y);
}

template<>
void Device::copy(const uint_tp n, const char* x, vptr<char> y);
template<>
void Device::copy(const uint_tp n, const half_fp* x,
                  vptr<half_fp> y);
template<>
void Device::copy(const uint_tp n, const float* x, vptr<float> y);
template<>
void Device::copy(const uint_tp n, const double* x, vptr<double> y);
template<>
void Device::copy(const uint_tp n, const int8_t* x, vptr<int8_t> y);
template<>
void Device::copy(const uint_tp n, const int16_t* x, vptr<int16_t> y);
template<>
void Device::copy(const uint_tp n, const int32_t* x, vptr<int32_t> y);
template<>
void Device::copy(const uint_tp n, const int64_t* x, vptr<int64_t> y);
template<>
void Device::copy(const uint_tp n, const uint8_t* x, vptr<uint8_t> y);
template<>
void Device::copy(const uint_tp n, const uint16_t* x, vptr<uint16_t> y);
template<>
void Device::copy(const uint_tp n, const uint32_t* x, vptr<uint32_t> y);
template<>
void Device::copy(const uint_tp n, const uint64_t* x, vptr<uint64_t> y);
template<>
void Device::copy(const uint_tp n, const void* x, vptr<void> y);


template<typename Dtype>
void Device::copy(const uint_tp n, vptr<const Dtype> x, Dtype* y) {
  this->memcpy(safe_sizeof<Dtype>() * n, x, y);
}

template<>
void Device::copy(const uint_tp n, vptr<const half_fp> x,
                  half_fp* y);
template<>
void Device::copy(const uint_tp n, vptr<const float> x, float* y);
template<>
void Device::copy(const uint_tp n, vptr<const double> x, double* y);
template<>
void Device::copy(const uint_tp n, vptr<const int8_t> x, int8_t* y);
template<>
void Device::copy(const uint_tp n, vptr<const int16_t> x, int16_t* y);
template<>
void Device::copy(const uint_tp n, vptr<const int32_t> x, int32_t* y);
template<>
void Device::copy(const uint_tp n, vptr<const int64_t> x, int64_t* y);
template<>
void Device::copy(const uint_tp n, vptr<const uint8_t> x, uint8_t* y);
template<>
void Device::copy(const uint_tp n, vptr<const uint16_t> x, uint16_t* y);
template<>
void Device::copy(const uint_tp n, vptr<const uint32_t> x, uint32_t* y);
template<>
void Device::copy(const uint_tp n, vptr<const uint64_t> x, uint64_t* y);
template<>
void Device::copy(const uint_tp n, vptr<const void> x, void* y);


template<>
void Device::gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                  const uint_tp m, const uint_tp n, const uint_tp k,
                  const half_fp alpha, vptr<const half_fp> a,
                  vptr<const half_fp> b,
                  const half_fp beta, vptr<half_fp> c) {
  this->gemm_half(trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

template<>
void Device::gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                  const uint_tp m, const uint_tp n, const uint_tp k,
                  const float alpha, vptr<const float> a,
                  vptr<const float> b, const float beta, vptr<float> c) {
  this->gemm_float(trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

template<>
void Device::gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                  const uint_tp m, const uint_tp n, const uint_tp k,
                  const double alpha, vptr<const double> a,
                  vptr<const double> b, const double beta, vptr<double> c) {
  this->gemm_double(trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

template<>
void Device::gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                  const uint_tp n, const half_fp alpha,
                  vptr<const half_fp> a,
                  vptr<const half_fp> x, const half_fp beta,
                  vptr<half_fp> y) {
  this->gemv_half(trans_a, m, n, alpha, a, x, beta, y);
}

template<>
void Device::gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                  const uint_tp n, const float alpha,
                  vptr<const float> a,
                  vptr<const float> x, const float beta,
                  vptr<float> y) {
  this->gemv_float(trans_a, m, n, alpha, a, x, beta, y);
}

template<>
void Device::gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                  const uint_tp n, const double alpha,
                  vptr<const double> a,
                  vptr<const double> x, const double beta,
                  vptr<double> y) {
  this->gemv_double(trans_a, m, n, alpha, a, x, beta, y);
}

template<>
void Device::axpy(const uint_tp n, const half_fp alpha,
                  vptr<const half_fp> x, vptr<half_fp> y) {
  this->axpy_half(n, alpha, x, y);
}

template<>
void Device::axpy(const uint_tp n, const float alpha,
                  vptr<const float> x, vptr<float> y) {
  this->axpy_float(n, alpha, x, y);
}

template<>
void Device::axpy(const uint_tp n, const double alpha,
                  vptr<const double> x, vptr<double> y) {
  this->axpy_double(n, alpha, x, y);
}

template<>
void Device::axpby(const uint_tp n, const half_fp alpha,
                   vptr<const half_fp> x,
                   const half_fp beta, vptr<half_fp> y) {
  this->axpby_half(n, alpha, x, beta, y);
}

template<>
void Device::axpby(const uint_tp n, const float alpha, vptr<const float> x,
                   const float beta, vptr<float> y) {
  this->axpby_float(n, alpha, x, beta, y);
}

template<>
void Device::axpby(const uint_tp n, const double alpha,
                   vptr<const double> x,
                   const double beta, vptr<double> y) {
  this->axpby_double(n, alpha, x, beta, y);
}

template<>
void Device::rng_uniform(const uint_tp n, const half_fp a,
                      const half_fp b, vptr<half_fp> r) {
  this->rng_uniform_half(n, a, b, r);
}

template<>
void Device::rng_uniform(const uint_tp n, const float a, const float b,
                               vptr<float> r) {
  this->rng_uniform_float(n, a, b, r);
}

template<>
void Device::rng_uniform(const uint_tp n, const double a,
                                const double b, vptr<double> r) {
  this->rng_uniform_double(n, a, b, r);
}

template<>
void Device::rng_gaussian(const uint_tp n, const half_fp mu,
                  const half_fp sigma, vptr<half_fp> r) {
  this->rng_gaussian_half(n, mu, sigma, r);
}

template<>
void Device::rng_gaussian(const uint_tp n, const float mu,
                                const float sigma, vptr<float> r) {
  this->rng_gaussian_float(n, mu, sigma, r);
}

template<>
void Device::rng_gaussian(const uint_tp n, const double mu,
                                 const double sigma, vptr<double> r) {
  this->rng_gaussian_double(n, mu, sigma, r);
}

template<>
void Device::rng_bernoulli(const uint_tp n, const half_fp p,
                           vptr<int> r) {
  this->rng_bernoulli_half(n, p, r);
}

template<>
void Device::rng_bernoulli(const uint_tp n, const float p, vptr<int> r) {
  this->rng_bernoulli_float(n, p, r);
}

template<>
void Device::rng_bernoulli(const uint_tp n, const double p, vptr<int> r) {
  this->rng_bernoulli_double(n, p, r);
}

template<>
void Device::rng_bernoulli(const uint_tp n, const half_fp p,
                           vptr<unsigned int> r) {
  this->rng_bernoulli_half(n, p, r);
}

template<>
void Device::rng_bernoulli(const uint_tp n, const float p,
                           vptr<unsigned int> r) {
  this->rng_bernoulli_float(n, p, r);
}

template<>
void Device::rng_bernoulli(const uint_tp n, const double p,
                           vptr<unsigned int> r) {
  this->rng_bernoulli_double(n, p, r);
}

template<>
void Device::dot(const uint_tp n, vptr<const half_fp> x,
                    vptr<const half_fp> y, half_fp *out) {
  this->dot_half(n, x, y, out);
}

template<>
void Device::dot(const uint_tp n, vptr<const float> x, vptr<const float> y,
                 float *out) {
  this->dot_float(n, x, y, out);
}

template<>
void Device::dot(const uint_tp n, vptr<const double> x, vptr<const double> y,
                 double *out) {
  this->dot_double(n, x, y, out);
}

template<>
void Device::asum(const uint_tp n, vptr<const half_fp> x,
                       half_fp* y) {
  this->asum_half(n, x, y);
}

template<>
void Device::asum(const uint_tp n, vptr<const float> x, float* y) {
  this->asum_float(n, x, y);
}

template<>
void Device::asum(const uint_tp n, vptr<const double> x, double* y) {
  this->asum_double(n, x, y);
}

template<>
void Device::scal(const uint_tp n, const half_fp alpha,
                       vptr<half_fp> x) {
  this->scal_half(n, alpha, x);
}

template<>
void Device::scal(const uint_tp n, const float alpha, vptr<float> x) {
  this->scal_float(n, alpha, x);

}

template<>
void Device::scal(const uint_tp n, const double alpha, vptr<double> x) {
  this->scal_double(n, alpha, x);
}

template<>
void Device::scale(const uint_tp n, const half_fp alpha,
                   vptr<const half_fp> x, vptr<half_fp> y) {
  this->scale_half(n, alpha, x, y);
}

template<>
void Device::scale(const uint_tp n, const float alpha, vptr<const float> x,
                   vptr<float> y) {
  this->scale_float(n, alpha, x, y);
}

template<>
void Device::scale(const uint_tp n, const double alpha,
                   vptr<const double> x, vptr<double> y) {
  this->scale_double(n, alpha, x, y);
}

}  // namespace caffe
