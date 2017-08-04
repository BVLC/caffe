#include "caffe/backend/device.hpp"

namespace caffe {

template<>
void device::gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const uint_tp M, const uint_tp N, const uint_tp K,
                  const half_float::half alpha, vptr<half_float::half> A,
                  vptr<half_float::half> B,
                  const half_float::half beta, vptr<half_float::half> C) {
  this->gemm_half(TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template<>
void device::gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const uint_tp M, const uint_tp N, const uint_tp K,
                  const float alpha, vptr<float> A,
                  vptr<float> B, const float beta, vptr<float> C) {
  this->gemm_float(TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template<>
void device::gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const uint_tp M, const uint_tp N, const uint_tp K,
                  const double alpha, vptr<double> A,
                  vptr<double> B, const double beta, vptr<double> C) {
  this->gemm_double(TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template<>
void device::gemv(const CBLAS_TRANSPOSE TransA, const uint_tp M,
                  const uint_tp N, const half_float::half alpha,
                  vptr<half_float::half> A,
                  vptr<half_float::half> x, const half_float::half beta,
                  vptr<half_float::half> y) {
  this->gemv_half(TransA, M, N, alpha, A, x, beta, y);
}

template<>
void device::gemv(const CBLAS_TRANSPOSE TransA, const uint_tp M,
                  const uint_tp N, const float alpha,
                  vptr<float> A,
                  vptr<float> x, const float beta,
                  vptr<float> y) {
  this->gemv_float(TransA, M, N, alpha, A, x, beta, y);
}

template<>
void device::gemv(const CBLAS_TRANSPOSE TransA, const uint_tp M,
                  const uint_tp N, const double alpha,
                  vptr<double> A,
                  vptr<double> x, const double beta,
                  vptr<double> y) {
  this->gemv_double(TransA, M, N, alpha, A, x, beta, y);
}

template<>
void device::axpy(const uint_tp N, const half_float::half alpha,
                  vptr<half_float::half> X, vptr<half_float::half> Y) {
  this->axpy_half(N, alpha, X, Y);
}

template<>
void device::axpy(const uint_tp N, const float alpha,
                  vptr<float> X, vptr<float> Y) {
  this->axpy_float(N, alpha, X, Y);
}

template<>
void device::axpy(const uint_tp N, const double alpha,
                  vptr<double> X, vptr<double> Y) {
  this->axpy_double(N, alpha, X, Y);
}

template<>
void device::axpby(const uint_tp N, const half_float::half alpha,
                   vptr<half_float::half> X,
                   const half_float::half beta, vptr<half_float::half> Y) {
  this->axpby_half(N, alpha, X, beta, Y);
}

template<>
void device::axpby(const uint_tp N, const float alpha, vptr<float> X,
                   const float beta, vptr<float> Y) {
  this->axpby_float(N, alpha, X, beta, Y);
}

template<>
void device::axpby(const uint_tp N, const double alpha,
                   vptr<double> X,
                   const double beta, vptr<double> Y) {
  this->axpby_double(N, alpha, X, beta, Y);
}

template<>
void device::set(const uint_tp N, const half_float::half alpha,
         vptr<half_float::half> X) {
  this->set_half(N, alpha, X);
}

template<>
void device::set(const uint_tp N, const float alpha, vptr<float> X) {
  this->set_float(N, alpha, X);
}

template<>
void device::set(const uint_tp N, const double alpha, vptr<double> X) {
  this->set_double(N, alpha, X);
}

template<>
void device::add_scalar(const uint_tp N, const half_float::half alpha,
                        vptr<half_float::half> X) {
  this->add_scalar_half(N, alpha, X);
}

template<>
void device::add_scalar(const uint_tp N, const float alpha, vptr<float> X) {
  this->add_scalar_float(N, alpha, X);
}

template<>
void device::add_scalar(const uint_tp N, const double alpha, vptr<double> X) {
  this->add_scalar_double(N, alpha, X);
}

template<>
void device::scal(const uint_tp N, const half_float::half alpha,
                  vptr<half_float::half> X) {
  this->scal_half(N, alpha, X);
}

template<>
void device::scal(const uint_tp N, const float alpha, vptr<float> X) {
  this->scal_float(N, alpha, X);
}

template<>
void device::scal(const uint_tp N, const double alpha, vptr<double> X) {
  this->scal_double(N, alpha, X);
}

template<>
void device::add(const uint_tp N, vptr<half_float::half> a,
                      vptr<half_float::half> b, vptr<half_float::half> y) {
  this->add_half(N, a, b, y);
}

template<>
void device::add(const uint_tp N, vptr<float> a,
                       vptr<float> b, vptr<float> y) {
  this->add_float(N, a, b, y);
}

template<>
void device::add(const uint_tp N, vptr<double> a,
                       vptr<double> b, vptr<double> y) {
  this->add_double(N, a, b, y);
}

template<>
void device::sub(const uint_tp N, vptr<half_float::half> a,
                      vptr<half_float::half> b, vptr<half_float::half> y) {
  this->sub_half(N, a, b, y);
}

template<>
void device::sub(const uint_tp N, vptr<float> a, vptr<float> b,
                       vptr<float> y) {
  this->sub_float(N, a, b, y);
}

template<>
void device::sub(const uint_tp N, vptr<double> a, vptr<double> b,
                        vptr<double> y) {
  this->sub_double(N, a, b, y);
}

template<>
void device::mul(const uint_tp N, vptr<half_float::half> a,
                      vptr<half_float::half> b, vptr<half_float::half> y) {
  this->mul_half(N, a, b, y);
}

template<>
void device::mul(const uint_tp N, vptr<float> a,
                      vptr<float> b, vptr<float> y) {
  this->mul_float(N, a, b, y);
}

template<>
void device::mul(const uint_tp N, vptr<double> a,
                      vptr<double> b, vptr<double> y) {
  this->mul_double(N, a, b, y);
}

template<>
void device::div(const uint_tp N, vptr<half_float::half> a,
                      vptr<half_float::half> b, vptr<half_float::half> y) {
  this->div_half(N, a, b, y);
}

template<>
void device::div(const uint_tp N, vptr<float> a, vptr<float> b, vptr<float> y) {
  this->div_float(N, a, b, y);
}

template<>
void device::div(const uint_tp N, vptr<double> a, vptr<double> b,
                 vptr<double> y) {
  this->div_double(N, a, b, y);
}

template<>
void device::abs(const uint_tp n, vptr<half_float::half> a,
                 vptr<half_float::half> y) {
  this->abs_half(n, a, y);
}

template<>
void device::abs(const uint_tp n, vptr<float> a, vptr<float> y) {
  this->abs_float(n, a, y);
}

template<>
void device::abs(const uint_tp n, vptr<double> a, vptr<double> y) {
  this->abs_double(n, a, y);
}

template<>
void device::exp(const uint_tp n, vptr<half_float::half> a,
                      vptr<half_float::half> y) {
  this->exp_half(n, a, y);
}

template<>
void device::exp(const uint_tp n, vptr<float> a, vptr<float> y) {
  this->exp_float(n, a, y);
}

template<>
void device::exp(const uint_tp n, vptr<double> a, vptr<double> y) {
  this->exp_double(n, a, y);
}

template<>
void device::log(const uint_tp n, vptr<half_float::half> a,
                      vptr<half_float::half> y) {
  this->log_half(n, a, y);
}

template<>
void device::log(const uint_tp n, vptr<float> a, vptr<float> y) {
  this->log_float(n, a, y);
}

template<>
void device::log(const uint_tp n, vptr<double> a, vptr<double> y) {
  this->log_double(n, a, y);
}

template<>
void device::powx(const uint_tp n, vptr<half_float::half> a,
                       const half_float::half b,
                       vptr<half_float::half> y) {
  this->powx_half(n, a, b, y);
}

template<>
void device::powx(const uint_tp n, vptr<float> a, const float b,
                        vptr<float> y) {
  this->powx_float(n, a, b, y);
}

template<>
void device::powx(const uint_tp n, vptr<double> a, const double b,
                         vptr<double> y) {
  this->powx_double(n, a, b, y);
}

template<>
void device::sqrt(const uint_tp n, vptr<half_float::half> a,
                       vptr<half_float::half> y) {
  this->sqrt_half(n, a, y);
}

template<>
void device::sqrt(const uint_tp n, vptr<float> a, vptr<float> y) {
  this->sqrt_float(n, a, y);
}

template<>
void device::sqrt(const uint_tp n, vptr<double> a, vptr<double> y) {
  this->sqrt_double(n, a, y);
}


template<>
void device::rng_uniform(const uint_tp n, const half_float::half a,
                      const half_float::half b, vptr<half_float::half> r) {
  this->rng_uniform_half(n, a, b, r);
}

template<>
void device::rng_uniform(const uint_tp n, const float a, const float b,
                               vptr<float> r) {
  this->rng_uniform_float(n, a, b, r);
}

template<>
void device::rng_uniform(const uint_tp n, const double a,
                                const double b, vptr<double> r) {
  this->rng_uniform_double(n, a, b, r);
}

template<>
void device::rng_gaussian(const uint_tp n, const half_float::half mu,
                  const half_float::half sigma, vptr<half_float::half> r) {
  this->rng_gaussian_half(n, mu, sigma, r);
}

template<>
void device::rng_gaussian(const uint_tp n, const float mu,
                                const float sigma, vptr<float> r) {
  this->rng_gaussian_float(n, mu, sigma, r);
}

template<>
void device::rng_gaussian(const uint_tp n, const double mu,
                                 const double sigma, vptr<double> r) {
  this->rng_gaussian_double(n, mu, sigma, r);
}

template<>
void device::rng_bernoulli(const uint_tp n, const half_float::half p,
                           vptr<int_tp> r) {
  this->rng_bernoulli_half(n, p, r);
}

template<>
void device::rng_bernoulli(const uint_tp n, const float p, vptr<int_tp> r) {
  this->rng_bernoulli_float(n, p, r);
}

template<>
void device::rng_bernoulli(const uint_tp n, const double p, vptr<int_tp> r) {
  this->rng_bernoulli_double(n, p, r);
}

template<>
void device::dot(const uint_tp n, vptr<half_float::half> x,
                    vptr<half_float::half> y, half_float::half *out) {
  this->dot_half(n, x, y, out);
}

template<>
void device::dot(const uint_tp n, vptr<float> x, vptr<float> y,
                 float *out) {
  this->dot_float(n, x, y, out);
}

template<>
void device::dot(const uint_tp n, vptr<double> x, vptr<double> y,
                 double *out) {
  this->dot_double(n, x, y, out);
}

template<>
void device::asum(const uint_tp n, vptr<half_float::half> x,
                       half_float::half* y) {
  this->asum_half(n, x, y);
}

template<>
void device::asum(const uint_tp n, vptr<float> x, float* y) {
  this->asum_float(n, x, y);
}

template<>
void device::asum(const uint_tp n, vptr<double> x, double* y) {
  this->asum_double(n, x, y);
}

template<>
void device::sign(const uint_tp n, vptr<half_float::half> x,
                       vptr<half_float::half> y) {
  this->sign_half(n, x, y);
}

template<>
void device::sign(const uint_tp n, vptr<float> x, vptr<float> y) {
  this->sign_float(n, x, y);
}

template<>
void device::sign(const uint_tp n, vptr<double> x, vptr<double> y) {
  this->sign_double(n, x, y);
}

template<>
void device::sgnbit(const uint_tp n, vptr<half_float::half> x,
                         vptr<half_float::half> y) {
  this->sgnbit_half(n, x, y);
}

template<>
void device::sgnbit(const uint_tp n, vptr<float> x, vptr<float> y) {
  this->sgnbit_float(n, x, y);
}

template<>
void device::sgnbit(const uint_tp n, vptr<double> x, vptr<double> y) {
  this->sgnbit_double(n, x, y);
}

template<>
void device::fabs(const uint_tp n, vptr<half_float::half> x,
                       vptr<half_float::half> y) {
  this->fabs_half(n, x, y);
}

template<>
void device::fabs(const uint_tp n, vptr<float> x, vptr<float> y) {
  this->fabs_float(n, x, y);
}

template<>
void device::fabs(const uint_tp n, vptr<double> x, vptr<double> y) {
  this->fabs_double(n, x, y);
}


template<>
void device::scale(const uint_tp n, const half_float::half alpha,
                   vptr<half_float::half> x, vptr<half_float::half> y) {
  this->scale_half(n, alpha, x, y);
}

template<>
void device::scale(const uint_tp n, const float alpha,
                         vptr<float> x, vptr<float> y) {
  this->scale_float(n, alpha, x, y);

}

template<>
void device::scale(const uint_tp n, const double alpha,
                          vptr<double> x, vptr<double> y) {
  this->scale_double(n, alpha, x, y);
}

}
