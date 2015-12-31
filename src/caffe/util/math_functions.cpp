#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, const int_tp M,
                           const int_tp N, const int_tp K, const float alpha,
                           const float* A, const float* B, const float beta,
                           float* C) {
  int_tp lda = (TransA == CblasNoTrans) ? K : M;
  int_tp ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB, const int_tp M,
                            const int_tp N, const int_tp K, const double alpha,
                            const double* A, const double* B, const double beta,
                            double* C) {
  int_tp lda = (TransA == CblasNoTrans) ? K : M;
  int_tp ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

template<>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int_tp M,
                           const int_tp N, const float alpha, const float* A,
                           const float* x, const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template<>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int_tp M,
                            const int_tp N, const double alpha, const double* A,
                            const double* x, const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template<>
void caffe_axpy<float>(const int_tp N, const float alpha, const float* X,
                       float* Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template<>
void caffe_axpy<double>(const int_tp N, const double alpha, const double* X,
                        double* Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

template<typename Dtype>
void caffe_set(const int_tp N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int_tp i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int32_t>(const int_tp N, const int alpha, int* Y);
template void caffe_set<uint32_t>(const int_tp N, const uint32_t alpha,
                                  uint32_t* Y);
template void caffe_set<int64_t>(const int_tp N, int64_t alpha, int64_t* Y);
template void caffe_set<uint64_t>(const int_tp N, const uint64_t alpha,
                                  uint64_t* Y);
template void caffe_set<float>(const int_tp N, const float alpha, float* Y);
template void caffe_set<double>(const int_tp N, const double alpha, double* Y);

template<>
void caffe_add_scalar(const int_tp N, const float alpha, float* Y) {
  for (int_tp i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template<>
void caffe_add_scalar(const int_tp N, const double alpha, double* Y) {
  for (int_tp i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template<typename Dtype>
void caffe_cpu_copy(const int_tp N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
  }
}

template void caffe_cpu_copy<int_tp>(const int_tp N, const int_tp* X,
                                     int_tp* Y);
template void caffe_cpu_copy<uint_tp>(const int_tp N, const uint_tp* X,
uint_tp* Y);
template void caffe_cpu_copy<float>(const int_tp N, const float* X, float* Y);
template void caffe_cpu_copy<double>(const int_tp N, const double* X,
                                     double* Y);

template<typename Dtype>
void caffe_copy(const int_tp N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
#ifdef USE_CUDA
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#endif  // USE_CUDA
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int_tp>(const int_tp N, const int_tp* X, int_tp* Y);
template void caffe_copy<uint_tp>(const int_tp N, const uint_tp* X,
uint_tp* Y);
template void caffe_copy<float>(const int_tp N, const float* X, float* Y);
template void caffe_copy<double>(const int_tp N, const double* X, double* Y);

template<>
void caffe_scal<float>(const int_tp N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template<>
void caffe_scal<double>(const int_tp N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template<>
void caffe_cpu_axpby<float>(const int_tp N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template<>
void caffe_cpu_axpby<double>(const int_tp N, const double alpha,
                             const double* X, const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template<>
void caffe_add<float>(const int_tp n, const float* a, const float* b,
                      float* y) {
  vsAdd(n, a, b, y);
}

template<>
void caffe_add<double>(const int_tp n, const double* a, const double* b,
                       double* y) {
  vdAdd(n, a, b, y);
}

template<>
void caffe_sub<float>(const int_tp n, const float* a, const float* b,
                      float* y) {
  vsSub(n, a, b, y);
}

template<>
void caffe_sub<double>(const int_tp n, const double* a, const double* b,
                       double* y) {
  vdSub(n, a, b, y);
}

template<>
void caffe_mul<float>(const int_tp n, const float* a, const float* b,
                      float* y) {
  vsMul(n, a, b, y);
}

template<>
void caffe_mul<double>(const int_tp n, const double* a, const double* b,
                       double* y) {
  vdMul(n, a, b, y);
}

template<>
void caffe_div<float>(const int_tp n, const float* a, const float* b,
                      float* y) {
  vsDiv(n, a, b, y);
}

template<>
void caffe_div<double>(const int_tp n, const double* a, const double* b,
                       double* y) {
  vdDiv(n, a, b, y);
}

template<>
void caffe_powx<float>(const int_tp n, const float* a, const float b,
                       float* y) {
  vsPowx(n, a, b, y);
}

template<>
void caffe_powx<double>(const int_tp n, const double* a, const double b,
                        double* y) {
  vdPowx(n, a, b, y);
}

template<>
void caffe_sqr<float>(const int_tp n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template<>
void caffe_sqr<double>(const int_tp n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template<>
void caffe_exp<float>(const int_tp n, const float* a, float* y) {
  vsExp(n, a, y);
}

template<>
void caffe_exp<double>(const int_tp n, const double* a, double* y) {
  vdExp(n, a, y);
}

template<>
void caffe_log<float>(const int_tp n, const float* a, float* y) {
  vsLn(n, a, y);
}

template<>
void caffe_log<double>(const int_tp n, const double* a, double* y) {
  vdLn(n, a, y);
}

template<>
void caffe_abs<float>(const int_tp n, const float* a, float* y) {
  vsAbs(n, a, y);
}

template<>
void caffe_abs<double>(const int_tp n, const double* a, double* y) {
  vdAbs(n, a, y);
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

void caffe_rng_uniform(const int_tp n, uint_tp* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  boost::uniform_int<int_tp> random_distribution(
      std::numeric_limits<int_tp>::min(), std::numeric_limits<int_tp>::max());
  boost::variate_generator<caffe::rng_t*,
  boost::uniform_int<int_tp>> variate_generator(
      caffe_rng(), random_distribution);
  for (int_tp i = 0; i < n; ++i) {
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
float caffe_cpu_strided_dot<float>(const int_tp n, const float* x,
                                   const int_tp incx, const float* y,
                                   const int_tp incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template<>
double caffe_cpu_strided_dot<double>(const int_tp n, const double* x,
                                     const int_tp incx, const double* y,
                                     const int_tp incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template<typename Dtype>
Dtype caffe_cpu_dot(const int_tp n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int_tp n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int_tp n, const double* x, const double* y);

template<>
float caffe_cpu_asum<float>(const int_tp n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template<>
double caffe_cpu_asum<double>(const int_tp n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template<>
void caffe_cpu_scale<float>(const int_tp n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template<>
void caffe_cpu_scale<double>(const int_tp n, const double alpha,
                             const double *x, double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

}  // namespace caffe
