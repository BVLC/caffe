// Copyright 2014 BVLC and contributors.

extern "C" {
#include <cblas.h>
}

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>
#include <cmath>
#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/devices/cpu.hpp"
#include "caffe/util/mkl_alternate.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void CPUDevice<float>::gemm(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB, const int M,
                            const int N, const int K, const float alpha,
                            const float* A, const float* B, const float beta,
                            float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void CPUDevice<double>::gemm(const CBLAS_TRANSPOSE TransA,
                             const CBLAS_TRANSPOSE TransB, const int M,
                             const int N, const int K, const double alpha,
                             const double* A, const double* B,
                             const double beta, double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void CPUDevice<int>::gemm(const CBLAS_TRANSPOSE TransA,
                          const CBLAS_TRANSPOSE TransB, const int M,
                          const int N, const int K, const int alpha,
                          const int* A, const int* B, const int beta, int* C) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::gemm(const CBLAS_TRANSPOSE TransA,
                                   const CBLAS_TRANSPOSE TransB, const int M,
                                   const int N, const int K,
                                   const unsigned int alpha,
                                   const unsigned int* A, const unsigned int* B,
                                   const unsigned int beta, unsigned int* C) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<float>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                            const int N, const float alpha, const float* A,
                            const float* x, const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template<>
void CPUDevice<double>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                             const int N, const double alpha, const double* A,
                             const double* x, const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template<>
void CPUDevice<int>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                          const int N, const int alpha, const int* A,
                          const int* x, const int beta, int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                                   const int N, const unsigned int alpha,
                                   const unsigned int* A, const unsigned int* x,
                                   const unsigned int beta, unsigned int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<float>::axpy(const int N, const float alpha, const float* X,
                            float* Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template<>
void CPUDevice<double>::axpy(const int N, const double alpha, const double* X,
                             double* Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

template<>
void CPUDevice<int>::axpy(const int N, const int alpha, const int* X, int* Y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::axpy(const int N, const unsigned int alpha,
                                   const unsigned int* X, unsigned int* Y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<float>::axpby(const int N, const float alpha, const float* X,
                             const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template<>
void CPUDevice<double>::axpby(const int N, const double alpha, const double* X,
                              const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template<>
void CPUDevice<int>::axpby(const int N, const int alpha, const int* X,
                           const int beta, int* Y) { NOT_IMPLEMENTED; }

template<>
void CPUDevice<unsigned int>::axpby(const int N, const unsigned int alpha,
                                    const unsigned int* X,
                                    const unsigned int beta, unsigned int* Y) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUDevice<Dtype>::copy(const int N, const Dtype *X, Dtype *Y) {
  memcpy(Y, X, sizeof(Dtype) * N);
}

template<typename Dtype>
void CPUDevice<Dtype>::set(const int N, const Dtype alpha, Dtype *X) {
  if (alpha == 0) {
    memset(X, 0, sizeof(Dtype) * N);
    return;
  }
  std::fill_n(X, N, alpha);
}

template<typename Dtype>
void CPUDevice<Dtype>::add_scalar(const int N, const Dtype alpha, Dtype *X) {
  for (int i = 0; i < N; ++i) {
    X[i] += alpha;
  }
}

template<>
void CPUDevice<float>::scal(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template<>
void CPUDevice<double>::scal(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template<>
void CPUDevice<int>::scal(const int N, const int alpha, int *X) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::scal(const int N, const unsigned int alpha,
                                   unsigned int *X) { NOT_IMPLEMENTED; }

template<>
void CPUDevice<float>::sqr(const int N, const float* a, float* y) {
  vsSqr(N, a, y);
}

template<>
void CPUDevice<double>::sqr(const int N, const double* a, double* y) {
  vdSqr(N, a, y);
}

template<>
void CPUDevice<int>::sqr(const int N, const int* a, int* y) { NOT_IMPLEMENTED; }

template<>
void CPUDevice<unsigned int>::sqr(const int N, const unsigned int* a,
                                  unsigned int* y) { NOT_IMPLEMENTED; }

template<>
void CPUDevice<float>::add(const int N, const float* a, const float* b,
                           float* y) {
  vsAdd(N, a, b, y);
}

template<>
void CPUDevice<double>::add(const int N, const double* a, const double* b,
                            double* y) {
  vdAdd(N, a, b, y);
}

template<>
void CPUDevice<int>::add(const int N, const int* a, const int* b, int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::add(const int N, const unsigned int* a,
                                  const unsigned int* b, unsigned int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<float>::sub(const int N, const float* a, const float* b,
                           float* y) {
  vsSub(N, a, b, y);
}

template<>
void CPUDevice<double>::sub(const int N, const double* a, const double* b,
                            double* y) {
  vdSub(N, a, b, y);
}

template<>
void CPUDevice<int>::sub(const int N, const int* a, const int* b, int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::sub(const int N, const unsigned int* a,
                                  const unsigned int* b, unsigned int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<float>::mul(const int N, const float* a, const float* b,
                           float* y) {
  vsMul(N, a, b, y);
}

template<>
void CPUDevice<double>::mul(const int N, const double* a, const double* b,
                            double* y) {
  vdMul(N, a, b, y);
}

template<>
void CPUDevice<int>::mul(const int N, const int* a, const int* b, int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::mul(const int N, const unsigned int* a,
                                  const unsigned int* b, unsigned int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<float>::div(const int N, const float* a, const float* b,
                           float* y) {
  vsDiv(N, a, b, y);
}

template<>
void CPUDevice<double>::div(const int N, const double* a, const double* b,
                            double* y) {
  vdDiv(N, a, b, y);
}

template<>
void CPUDevice<int>::div(const int N, const int* a, const int* b, int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::div(const int N, const unsigned int* a,
                                  const unsigned int* b, unsigned int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<float>::powx(const int N, const float* a, const float b,
                            float* y) {
  vsPowx(N, a, b, y);
}

template<>
void CPUDevice<double>::powx(const int N, const double* a, const double b,
                             double* y) {
  vdPowx(N, a, b, y);
}

template<>
void CPUDevice<int>::powx(const int N, const int* a, const int b, int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::powx(const int N, const unsigned int* a,
                                   const unsigned int b, unsigned int* y) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
static Dtype _nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template<typename Dtype>
void CPUDevice<Dtype>::rng_uniform(const int N, const Dtype a, const Dtype b,
                                   Dtype* r) {
  CHECK_GE(N, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, _nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < N; ++i) {
    r[i] = variate_generator();
  }
}

template <>
void CPUDevice<int>::rng_uniform(const int n, const int a, const int b,
                                 int* r) { NOT_IMPLEMENTED; }

template <>
void CPUDevice<unsigned int>::rng_uniform(const int n, const unsigned int a,
                                          const unsigned int b,
                                          unsigned int* r) { NOT_IMPLEMENTED; }

template<typename Dtype>
void CPUDevice<Dtype>::rng_gaussian(const int N, const Dtype mu,
                                    const Dtype sigma, Dtype* r) {
  CHECK_GE(N, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(mu, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < N; ++i) {
    r[i] = variate_generator();
  }
}

template <>
void CPUDevice<int>::rng_gaussian(const int n, const int mu, const int sigma,
                                  int* r) { NOT_IMPLEMENTED; }

template <>
void CPUDevice<unsigned int>::rng_gaussian(const int n, const unsigned int mu,
                                           const unsigned int sigma,
                                           unsigned int* r) { NOT_IMPLEMENTED; }

template<typename Dtype>
void CPUDevice<Dtype>::rng_bernoulli(const int N, const Dtype p, int* r) {
  CHECK_GE(N, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < N; ++i) {
    r[i] = variate_generator();
  }
}

template<>
void CPUDevice<int>::rng_bernoulli(const int N, const int p, int* r) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::rng_bernoulli(const int N, const unsigned int p,
                                            int* r) { NOT_IMPLEMENTED; }

template<typename Dtype>
void CPUDevice<Dtype>::rng_bernoulli(const int N, const Dtype p,
                                     unsigned int* r) {
  CHECK_GE(N, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < N; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template<>
void CPUDevice<int>::rng_bernoulli(const int N, const int p, unsigned int* r) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::rng_bernoulli(const int N, const unsigned int p,
                                            unsigned int* r) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<float>::exp(const int N, const float* a, float* y) {
  vsExp(N, a, y);
}

template<>
void CPUDevice<double>::exp(const int N, const double* a, double* y) {
  vdExp(N, a, y);
}

template<>
void CPUDevice<int>::exp(const int N, const int* a, int* y) { NOT_IMPLEMENTED; }

template<>
void CPUDevice<unsigned int>::exp(const int N, const unsigned int* a,
                                  unsigned int* y) { NOT_IMPLEMENTED; }

template<>
void CPUDevice<float>::dot(const int N, const float* x, const float* y,
                           float* out) {
  *out =  cblas_sdot(N, x, 1, y, 1);
}

template<>
void CPUDevice<double>::dot(const int N, const double* x, const double* y,
                            double* out) {
  *out = cblas_ddot(N, x, 1, y, 1);
}

template<>
void CPUDevice<int>::dot(const int N, const int* x, const int* y, int* out) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::dot(const int N, const unsigned int* x,
                                  const unsigned int* y, unsigned int* out) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<float>::hamming_distance(const int N, const float* x,
                                        const float* y, int* out) {
  int dist = 0;
  for (int i = 0; i < N; ++i) {
    dist += __builtin_popcount(static_cast<uint32_t>(x[i]) ^
                               static_cast<uint32_t>(y[i]));
  }
  *out = dist;
}

template<>
void CPUDevice<double>::hamming_distance(const int N, const double* x,
                                         const double* y, int* out) {
  int dist = 0;
  for (int i = 0; i < N; ++i) {
    dist += __builtin_popcountl(static_cast<uint64_t>(x[i]) ^
                                static_cast<uint64_t>(y[i]));
  }
  *out = dist;
}

template<>
void CPUDevice<int>::hamming_distance(const int N, const int* x, const int* y,
                                      int* out) { NOT_IMPLEMENTED; }

template<>
void CPUDevice<unsigned int>::hamming_distance(const int N,
                                               const unsigned int* x,
                                               const unsigned int* y,
                                               int* out) { NOT_IMPLEMENTED; }

// Returns the sum of the absolute values of the elements of vector x
template<>
void CPUDevice<float>::asum(const int N, const float* x, float* y) {
  *y = cblas_sasum(N, x, 1);
}

template<>
void CPUDevice<double>::asum(const int N, const double* x, double* y) {
  *y = cblas_dasum(N, x, 1);
}

template<>
void CPUDevice<int>::asum(const int N, const int* x, int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::asum(const int N, const unsigned int* x,
                                   unsigned int* y) { NOT_IMPLEMENTED; }

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template<typename Dtype>
static inline char _sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

template<typename Dtype>
void CPUDevice<Dtype>::sign(const int N, const Dtype* x, Dtype* y) {
  CHECK_GT(N, 0);
  CHECK(x);
  CHECK(y);
  for (int i = 0; i < N; ++i) {
    y[i] = _sign<Dtype>(x[i]);
  }
}

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro
template<typename Dtype>
void CPUDevice<Dtype>::sgnbit(const int N, const Dtype* x, Dtype* y) {
  CHECK_GT(N, 0);
  CHECK(x);
  CHECK(y);
  for (int i = 0; i < N; ++i) {
    y[i] = std::signbit(x[i]);
  }
}

template<typename Dtype>
void CPUDevice<Dtype>::fabs(const int N, const Dtype* x, Dtype* y) {
  CHECK_GT(N, 0);
  CHECK(x);
  CHECK(y);
  for (int i = 0; i < N; ++i) {
    y[i] = std::fabs(x[i]);
  }
}

template<>
void CPUDevice<float>::scale(const int N, const float alpha, const float *x,
                             float* y) {
  cblas_scopy(N, x, 1, y, 1);
  cblas_sscal(N, alpha, y, 1);
}

template<>
void CPUDevice<double>::scale(const int N, const double alpha, const double *x,
                              double* y) {
  cblas_dcopy(N, x, 1, y, 1);
  cblas_dscal(N, alpha, y, 1);
}

template<>
void CPUDevice<int>::scale(const int N, const int alpha, const int *x, int* y) {
  NOT_IMPLEMENTED;
}

template<>
void CPUDevice<unsigned int>::scale(const int N, const unsigned int alpha,
                                    const unsigned int *x, unsigned int* y) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void CPUDevice<Dtype>::im2col(const Dtype* data_im, const int channels,
                              const int height, const int width,
                              const int ksize, const int pad, const int stride,
                              Dtype* data_col) {
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride - pad + h_offset;
        int w_pad = w * stride - pad + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

template<typename Dtype>
void CPUDevice<Dtype>::col2im(const Dtype* data_col, const int channels,
                              const int height, const int width,
                              const int ksize, const int pad, const int stride,
                              Dtype* data_im) {
  memset(data_im, 0, sizeof(Dtype) * height * width * channels);
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride - pad + h_offset;
        int w_pad = w * stride - pad + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

INSTANTIATE_CLASS(CPUDevice);
template class CPUDevice<int>;
template class CPUDevice<unsigned int>;

}  // namespace caffe
