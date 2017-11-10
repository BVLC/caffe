#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

#ifdef USE_GPU_HALF
template<>
void caffe_add_scalar(const int_tp n, const half_fp alpha,
                      half_fp* Y) {
  for (int_tp i = 0; i < n; ++i) {
    Y[i] += alpha;
  }
}

template<>
void caffe_cpu_gemm<half_fp>(const CBLAS_TRANSPOSE trans_a,
                          const CBLAS_TRANSPOSE trans_b, const int_tp m,
                          const int_tp n, const int_tp k,
                          const half_fp alpha,
                          const half_fp* a, const half_fp* b,
                          const half_fp beta,
                          half_fp* c) {
  int_tp inc_a = (trans_a == CblasNoTrans) ? 1 : m;
  int_tp inc_b = (trans_b == CblasNoTrans) ? n : 1;
  for (int_tp m = 0; m < m; m++) {
    for (int_tp n = 0; n < n; n++) {
      half_fp acc = 0;
      int_tp b_index = trans_b == CblasNoTrans ?
                       n : k * n;
      int_tp a_index = trans_a == CblasNoTrans ?
                       k * m : m;
      for (int_tp k = 0; k < k; k++) {
        acc += a[a_index] * b[b_index];
        a_index += inc_a;
        b_index += inc_b;
      }
      if (beta != 0)
        c[m * n + n] = acc * alpha + beta * c[m * n + n];
      else
        c[m * n + n] = acc * alpha;
    }
  }
}

template<>
void caffe_cpu_gemv<half_fp>(const CBLAS_TRANSPOSE trans_a,
                   const int_tp m, const int_tp n, const half_fp alpha,
                   const half_fp* a, const half_fp* X,
                   const half_fp beta, half_fp* Y) {
  int_tp a_inc = (trans_a == CblasNoTrans) ? 1 : n;
  int_tp y_cnt = (trans_a == CblasNoTrans) ? m : n;
  int_tp x_cnt = (trans_a == CblasNoTrans) ? n : m;
  for (int_tp m = 0; m < y_cnt; m++) {
    int_tp a_index = (trans_a == CblasNoTrans) ? m * n : m;
    half_fp acc = 0;
    for (int_tp n = 0; n < x_cnt; n++) {
      acc += a[a_index] * X[n];
      a_index += a_inc;
    }
    if (beta == 0)
      Y[m] = acc * alpha;
    else
      Y[m] = acc * alpha + beta * Y[m];
  }
}

template<>
void caffe_axpy<half_fp>(const int_tp n, const half_fp alpha,
                                  const half_fp* X,
                                  half_fp* Y) {
  for (int_tp n = 0; n < n; n++) {
    Y[n] += alpha * X[n];
  }
}

template<>
void caffe_scal<half_fp>(const int_tp n, const half_fp alpha,
                                  half_fp *X) {
  for (int_tp n = 0; n < n; n++)
    X[n] *= alpha;
}

template<>
void caffe_cpu_axpby<half_fp>(const int_tp n,
                        const half_fp alpha, const half_fp* X,
                        const half_fp beta, half_fp* Y) {
  cblas_haxpby(n, alpha, X, 1, beta, Y, 1);
}

void vhAdd(const int_tp n, const half_fp* a, const half_fp* b,
                     half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = a[i] + b[i];
  }
}

template<>
void caffe_add<half_fp>(const int_tp n, const half_fp* a,
                               const half_fp* b, half_fp* Y) {
  vhAdd(n, a, b, Y);
}

void vhSub(const int_tp n, const half_fp* a, const half_fp* b,
           half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = a[i] - b[i];
  }
}

template<>
void caffe_sub<half_fp>(const int_tp n, const half_fp* a,
                               const half_fp* b, half_fp* Y) {
  vhSub(n, a, b, Y);
}

void vhMul(const int_tp n, const half_fp* a, const half_fp* b,
           half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = a[i] * b[i];
  }
}

template<>
void caffe_mul<half_fp>(const int_tp n, const half_fp* a,
                               const half_fp* b, half_fp* Y) {
  vhMul(n, a, b, Y);
}

void vhDiv(const int_tp n, const half_fp* a, const half_fp* b,
           half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = a[i] / b[i];
  }
}

template<>
void caffe_div<half_fp>(const int_tp n, const half_fp* a,
                               const half_fp* b, half_fp* Y) {
  vhDiv(n, a, b, Y);
}

void vhPowx(const int_tp n, const half_fp*a, const half_fp b,
            half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = pow(a[i], b);
  }
}

template<>
void caffe_powx<half_fp>(const int_tp n, const half_fp* a,
                                const half_fp b, half_fp* Y) {
  vhPowx(n, a, b, Y);
}

void vhSqr(const int_tp n, const half_fp *a, half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = sqrt(a[i]);
  }
}

template<>
void caffe_sqr<half_fp>(const int_tp n, const half_fp* a,
                                 half_fp* Y) {
  vhSqr(n, a, Y);
}

void vhExp(const int_tp n, const half_fp* a, half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = exp(a[i]);
  }
}

template<>
void caffe_exp<half_fp>(const int_tp n, const half_fp* a,
                                 half_fp* Y) {
  vhExp(n, a, Y);
}

void vhLn(const int_tp n, const half_fp* a, half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = log(a[i]);
  }
}

template<>
void caffe_log<half_fp>(const int_tp n, const half_fp* a,
                                 half_fp* Y) {
  vhLn(n, a, Y);
}

void vhAbs(const int_tp n, const half_fp *a, half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = fabs(a[i]);
  }
}

template<>
void caffe_abs<half_fp>(const int_tp n, const half_fp* a,
                                 half_fp* Y) {
  vhAbs(n, a, Y);
}

void vsHqrt(const int_tp n, const half_fp* a, half_fp* Y) {
  for (int_tp i = 0; i < n; i++) {
    Y[i] = sqrt(a[i]);
  }
}
template <>
void caffe_sqrt<half_fp>(const int_tp n, const half_fp* a,
                                  half_fp* Y) {
  vsHqrt(n, a, Y);
}

template<>
void caffe_rng_uniform<half_fp>(const int_tp n,
                             const half_fp a, const half_fp b,
                             half_fp* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<float> random_distribution(static_cast<float>(a),
    caffe_nextafter<float>(static_cast<float>(b)));

  boost::variate_generator<caffe::rng_t*,
    boost::uniform_real<float>> variate_generator(
    caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_gaussian<half_fp>(const int_tp n,
                         const half_fp a, const half_fp sigma,
                         half_fp* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  float fsigma = sigma;
  float fa = a;
  boost::normal_distribution<float> random_distribution(fa, fsigma);
  boost::variate_generator<caffe::rng_t*,
    boost::normal_distribution<float>> variate_generator(
        caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void caffe_rng_bernoulli<half_fp, unsigned int>(const int_tp n,
                                    const half_fp p, unsigned int* r) {
  // FIXME
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  float f_p = p;
  boost::bernoulli_distribution<float> random_distribution(f_p);
  boost::variate_generator<caffe::rng_t*,
  boost::bernoulli_distribution<float>> variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}
template<>
void caffe_rng_bernoulli<half_fp, int>(const int_tp n,
                                             const half_fp p, int* r) {
  // FIXME
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  float f_p = p;
  boost::bernoulli_distribution<float> random_distribution(f_p);
  boost::variate_generator<caffe::rng_t*,
  boost::bernoulli_distribution<float>> variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = static_cast<int>(variate_generator());
  }
}

template<>
void caffe_cpu_scale<half_fp>(const int_tp n,
                        const half_fp alpha, const half_fp *X,
                        half_fp* Y) {
  for (int_tp i = 0; i < n; i++)
    Y[i] = X[i];
  caffe_scal(n, alpha, Y);
}

template<>
half_fp caffe_cpu_strided_dot<half_fp>(const int_tp n,
                                const half_fp* X, const int_tp incx,
                                const half_fp* Y, const int_tp incy) {
  half_fp sum = 0;
  for (int_tp i = 0; i < n; i++)
    sum += X[i * incx] * Y[i * incy];
  return sum;
}

template<>
half_fp caffe_cpu_asum<half_fp>(const int_tp n,
                                                  const half_fp* X) {
  half_fp sum = 0;
  for (int_tp i = 0; i < n; i++)
    sum += fabs(X[i]);
  return sum;
}
#endif

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE trans_a,
                           const CBLAS_TRANSPOSE trans_b, const int_tp m,
                           const int_tp n, const int_tp k, const float alpha,
                           const float* a, const float* b, const float beta,
                           float* c) {
  int_tp lda = (trans_a == CblasNoTrans) ? k : m;
  int_tp ldb = (trans_b == CblasNoTrans) ? n : k;
  cblas_sgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb,
              beta, c, n);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b, const int_tp m,
                            const int_tp n, const int_tp k, const double alpha,
                            const double* a, const double* b, const double beta,
                            double* c) {
  int_tp lda = (trans_a == CblasNoTrans) ? k : m;
  int_tp ldb = (trans_b == CblasNoTrans) ? n : k;
  cblas_dgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb,
              beta, c, n);
}

template<>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE trans_a, const int_tp m,
                           const int_tp n, const float alpha, const float* a,
                           const float* X, const float beta, float* Y) {
  cblas_sgemv(CblasRowMajor, trans_a, m, n, alpha, a, n, X, 1, beta, Y, 1);
}

template<>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE trans_a, const int_tp m,
                            const int_tp n, const double alpha, const double* a,
                            const double* X, const double beta, double* Y) {
  cblas_dgemv(CblasRowMajor, trans_a, m, n, alpha, a, n, X, 1, beta, Y, 1);
}


template<>
void caffe_axpy<float>(const int_tp n, const float alpha, const float* X,
                       float* Y) {
  cblas_saxpy(n, alpha, X, 1, Y, 1);
}

template<>
void caffe_axpy<double>(const int_tp n, const double alpha, const double* X,
                        double* Y) {
  cblas_daxpy(n, alpha, X, 1, Y, 1);
}

template<typename Dtype>
void caffe_set(const int_tp n, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * n);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int_tp i = 0; i < n; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int32_t>(const int_tp n, const int alpha, int* Y);
template void caffe_set<uint32_t>(const int_tp n, const uint32_t alpha,
                                  uint32_t* Y);
template void caffe_set<int64_t>(const int_tp n, int64_t alpha, int64_t* Y);
template void caffe_set<uint64_t>(const int_tp n, const uint64_t alpha,
                                  uint64_t* Y);
#ifdef USE_GPU_HALF
template void caffe_set<half_fp>(const int_tp n,
                             const half_fp alpha, half_fp* Y);
#endif
template void caffe_set<float>(const int_tp n, const float alpha, float* Y);
template void caffe_set<double>(const int_tp n, const double alpha, double* Y);

template<>
void caffe_add_scalar(const int_tp n, const float alpha, float* Y) {
  for (int_tp i = 0; i < n; ++i) {
    Y[i] += alpha;
  }
}

template<>
void caffe_add_scalar(const int_tp n, const double alpha, double* Y) {
  for (int_tp i = 0; i < n; ++i) {
    Y[i] += alpha;
  }
}

template<typename Dtype>
void caffe_cpu_copy(const int_tp n, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    memcpy(Y, X, sizeof(Dtype) * n);  // NOLINT(caffe/alt_fn)
  }
}

template void caffe_cpu_copy<int_tp>(const int_tp n, const int_tp* X,
                                     int_tp* Y);
template void caffe_cpu_copy<uint_tp>(const int_tp n, const uint_tp* X,
uint_tp* Y);
#ifdef USE_GPU_HALF
template void caffe_cpu_copy<half_fp>(const int_tp n,
                                const half_fp* X, half_fp* Y);
#endif
template void caffe_cpu_copy<float>(const int_tp n, const float* X, float* Y);
template void caffe_cpu_copy<double>(const int_tp n, const double* X,
                                     double* Y);

template<typename Dtype>
void caffe_copy(const int_tp n, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
#ifdef USE_CUDA
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * n, cudaMemcpyDefault));
#endif  // USE_CUDA
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * n);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int_tp>(const int_tp n, const int_tp* X, int_tp* Y);
template void caffe_copy<uint_tp>(const int_tp n, const uint_tp* X,
uint_tp* Y);
template void caffe_copy<half_fp>(const int_tp n,
                                const half_fp* X, half_fp* Y);
template void caffe_copy<float>(const int_tp n, const float* X, float* Y);
template void caffe_copy<double>(const int_tp n, const double* X, double* Y);

template<>
void caffe_scal<float>(const int_tp n, const float alpha, float *X) {
  cblas_sscal(n, alpha, X, 1);
}

template<>
void caffe_scal<double>(const int_tp n, const double alpha, double *X) {
  cblas_dscal(n, alpha, X, 1);
}
template<>
void caffe_cpu_axpby<float>(const int_tp n, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(n, alpha, X, 1, beta, Y, 1);
}

template<>
void caffe_cpu_axpby<double>(const int_tp n, const double alpha,
                             const double* X, const double beta, double* Y) {
  cblas_daxpby(n, alpha, X, 1, beta, Y, 1);
}


template<>
void caffe_add<float>(const int_tp n, const float* a, const float* b,
                      float* Y) {
  vsAdd(n, a, b, Y);
}

template<>
void caffe_add<double>(const int_tp n, const double* a, const double* b,
                       double* Y) {
  vdAdd(n, a, b, Y);
}

template<>
void caffe_sub<float>(const int_tp n, const float* a, const float* b,
                      float* Y) {
  vsSub(n, a, b, Y);
}

template<>
void caffe_sub<double>(const int_tp n, const double* a, const double* b,
                       double* Y) {
  vdSub(n, a, b, Y);
}

template<>
void caffe_mul<float>(const int_tp n, const float* a, const float* b,
                      float* Y) {
  vsMul(n, a, b, Y);
}

template<>
void caffe_mul<double>(const int_tp n, const double* a, const double* b,
                       double* Y) {
  vdMul(n, a, b, Y);
}

template<>
void caffe_div<float>(const int_tp n, const float* a, const float* b,
                      float* Y) {
  vsDiv(n, a, b, Y);
}

template<>
void caffe_div<double>(const int_tp n, const double* a, const double* b,
                       double* Y) {
  vdDiv(n, a, b, Y);
}

template<>
void caffe_powx<float>(const int_tp n, const float* a, const float b,
                       float* Y) {
  vsPowx(n, a, b, Y);
}

template<>
void caffe_powx<double>(const int_tp n, const double* a, const double b,
                        double* Y) {
  vdPowx(n, a, b, Y);
}

template<>
void caffe_sqr<float>(const int_tp n, const float* a, float* Y) {
  vsSqr(n, a, Y);
}

template<>
void caffe_sqr<double>(const int_tp n, const double* a, double* Y) {
  vdSqr(n, a, Y);
}

template <>
void caffe_sqrt<float>(const int_tp n, const float* a, float* Y) {
  vsSqrt(n, a, Y);
}

template <>
void caffe_sqrt<double>(const int_tp n, const double* a, double* Y) {
  vdSqrt(n, a, Y);
}

template <>
void caffe_exp<float>(const int_tp n, const float* a, float* Y) {
  vsExp(n, a, Y);
}

template<>
void caffe_exp<double>(const int_tp n, const double* a, double* Y) {
  vdExp(n, a, Y);
}

template<>
void caffe_log<float>(const int_tp n, const float* a, float* Y) {
  vsLn(n, a, Y);
}

template<>
void caffe_log<double>(const int_tp n, const double* a, double* Y) {
  vdLn(n, a, Y);
}

template<>
void caffe_abs<float>(const int_tp n, const float* a, float* Y) {
  vsAbs(n, a, Y);
}

template<>
void caffe_abs<double>(const int_tp n, const double* a, double* Y) {
  vdAbs(n, a, Y);
}

uint_tp caffe_rng_rand() {
  return (*caffe_rng())();
}

template<typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

void caffe_rng_uniform(const uint_tp n, uint32_t* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  boost::uniform_int<uint32_t> random_distribution(
      std::numeric_limits<uint32_t>::min(),
      std::numeric_limits<uint32_t>::max());
  boost::variate_generator<caffe::rng_t*, boost::uniform_int<uint32_t> >
                            variate_generator(caffe_rng(), random_distribution);
  for (uint_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

void caffe_rng_uniform(const uint_tp n, uint64_t* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  boost::uniform_int<uint64_t> random_distribution(
      std::numeric_limits<uint64_t>::min(),
      std::numeric_limits<uint64_t>::max());
  boost::variate_generator<caffe::rng_t*, boost::uniform_int<uint64_t> >
                            variate_generator(caffe_rng(), random_distribution);
  for (uint_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template<typename Dtype>
void caffe_rng_uniform(const int_tp n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*,
  boost::uniform_real<Dtype>> variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int_tp n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int_tp n, const double a, const double b,
                               double* r);

template<typename Dtype>
void caffe_rng_gaussian(const int_tp n, const Dtype a, const Dtype sigma,
                        Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*,
  boost::normal_distribution<Dtype>> variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int_tp n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int_tp n, const double mu,
                                const double sigma, double* r);

template<typename Dtype, typename Itype>
void caffe_rng_bernoulli(const int_tp n, const Dtype p, Itype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*,
  boost::bernoulli_distribution<Dtype>> variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
    r[i] = static_cast<Itype>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double, unsigned long>(const int_tp n, const double p,  // NOLINT
                                                unsigned long* r);  // NOLINT

template
void caffe_rng_bernoulli<float, unsigned long>(const int_tp n, const float p,  // NOLINT
                                               unsigned long* r);  // NOLINT

template
void caffe_rng_bernoulli<double, long>(const int_tp n, const double p, long* r);  // NOLINT

template
void caffe_rng_bernoulli<float, long>(const int_tp n, const float p, long* r);  // NOLINT

template
void caffe_rng_bernoulli<double, unsigned int>(const int_tp n, const double p,
                                               unsigned int* r);

template
void caffe_rng_bernoulli<float, unsigned int>(const int_tp n, const float p,
                                              unsigned int* r);

template
void caffe_rng_bernoulli<double, int>(const int_tp n, const double p, int* r);

template
void caffe_rng_bernoulli<float, int>(const int_tp n, const float p, int* r);

template<>
float caffe_cpu_strided_dot<float>(const int_tp n, const float* X,
                                   const int_tp incx, const float* Y,
                                   const int_tp incy) {
  return cblas_sdot(n, X, incx, Y, incy);
}

template<>
double caffe_cpu_strided_dot<double>(const int_tp n, const double* X,
                                     const int_tp incx, const double* Y,
                                     const int_tp incy) {
  return cblas_ddot(n, X, incx, Y, incy);
}

template<typename Dtype>
Dtype caffe_cpu_dot(const int_tp n, const Dtype* X, const Dtype* Y) {
  return caffe_cpu_strided_dot(n, X, 1, Y, 1);
}

#ifdef USE_GPU_HALF
template
half_fp caffe_cpu_dot<half_fp>(const int_tp n,
                          const half_fp* X, const half_fp* Y);
#endif

template
float caffe_cpu_dot<float>(const int_tp n, const float* X, const float* Y);

template
double caffe_cpu_dot<double>(const int_tp n, const double* X, const double* Y);


template<>
float caffe_cpu_asum<float>(const int_tp n, const float* X) {
  return cblas_sasum(n, X, 1);
}

template<>
double caffe_cpu_asum<double>(const int_tp n, const double* X) {
  return cblas_dasum(n, X, 1);
}

template<>
void caffe_cpu_scale<float>(const int_tp n, const float alpha, const float *X,
                            float* Y) {
  cblas_scopy(n, X, 1, Y, 1);
  cblas_sscal(n, alpha, Y, 1);
}

template<>
void caffe_cpu_scale<double>(const int_tp n, const double alpha,
                             const double *X, double* Y) {
  cblas_dcopy(n, X, 1, Y, 1);
  cblas_dscal(n, alpha, Y, 1);
}

}  // namespace caffe
