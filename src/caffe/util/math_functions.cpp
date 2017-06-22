/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#if USE_MKL
#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <algorithm>
#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/cpu_info.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const long N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const long N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const size_t N, const Dtype alpha, Dtype* Y) {
  // If we are executing parallel region already then do not start another one
  // if also number of data to be processed is smaller than arbitrary:
  // threashold 12*4 cachelines per thread then no parallelization is to be made
  #ifdef _OPENMP

  int nthr = omp_get_max_threads();
  int threshold = nthr * caffe::cpu::OpenMpManager::getProcessorSpeedMHz() / 3;
  bool run_parallel =  // Do not do parallel computation from non major threads
       caffe::cpu::OpenMpManager::isMajorThread(boost::this_thread::get_id());

  // Note: we Assume GPU's CPU path is single threaded
  if (omp_in_parallel() == 0) {
    // inactive parallel region may mean also batch 1,
    // but no new threads are to be created
    run_parallel = run_parallel && (Caffe::mode() != Caffe::GPU) &&
                   (N >= threshold);
  } else {
    // If we are running active parallel region then it is CPU
    run_parallel = run_parallel && (N >= threshold);
  }

  if (run_parallel) {
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
      Y[i] = alpha;
    }

    return;
  }

  #endif

  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
  } else {
    std::fill(Y, Y + N, alpha);
  }
}

template void caffe_set<char>(const size_t N, const char alpha, char* Y);
template void caffe_set<int>(const size_t N, const int alpha, int* Y);
template void caffe_set<float>(const size_t N, const float alpha, float* Y);
template void caffe_set<double>(const size_t N, const double alpha, double* Y);
template void caffe_set<size_t>(const size_t N, const size_t alpha, size_t* Y);

template <>
void caffe_add_scalar(const long N, const float alpha, float* Y) {
  for (long i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const long N, const double alpha, double* Y) {
  for (long i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_cpu_copy(const size_t N, const Dtype* X, Dtype* Y) {
  if (X == Y) return;

#ifdef _OPENMP
  static const int threshold = omp_get_max_threads() *
                          caffe::cpu::OpenMpManager::getProcessorSpeedMHz() / 3;
  const bool run_parallel =
    (N >= threshold) &&
    (omp_in_parallel() == 0) &&
    (Caffe::mode() != Caffe::GPU) &&
    (caffe::cpu::OpenMpManager::isMajorThread(boost::this_thread::get_id()));

  if (run_parallel) {
    const int block_mem_size = 256 * 1024;
    const int block_size = block_mem_size / sizeof(Dtype);
    #pragma omp parallel for
    for (size_t i = 0; i < N; i += block_size)
      memcpy(Y + i, X + i,
              (i + block_size > N) ? (N - i) * sizeof(Dtype) : block_mem_size);

    return;
  }
#endif

  memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
}

template void caffe_cpu_copy<int>(const size_t N, const int* X, int* Y);
template void caffe_cpu_copy<unsigned int>(const size_t N, const unsigned int* X,
    unsigned int* Y);
template void caffe_cpu_copy<float>(const size_t N, const float* X, float* Y);
template void caffe_cpu_copy<double>(const size_t N, const double* X, double* Y);

template <typename Dtype>
void caffe_copy(const size_t N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
#ifndef CPU_ONLY
    if (
#ifdef _OPENMP
         // If there are more than one openmp thread (we are in active region)
         // then checking Caffe::mode can create additional GPU Context
        (omp_in_parallel() == 0) &&
#endif
        (Caffe::mode() == Caffe::GPU)) {
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
    } else {
#endif
      caffe_cpu_copy<Dtype>(N, X, Y);
#ifndef CPU_ONLY
    }
#endif
  }
}

template void caffe_copy<bool>(const size_t N, const bool* X, bool* Y);
template void caffe_copy<int>(const size_t N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const size_t N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const size_t N, const float* X, float* Y);
template void caffe_copy<double>(const size_t N, const double* X, double* Y);
template void caffe_copy<char>(const size_t N, const char* X, char* Y);
template void caffe_copy<size_t>(const size_t N, const size_t* X, size_t* Y);

template <>
void caffe_scal<float>(const long N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const long N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_scal<size_t>(const long N, const size_t alpha, size_t *X) {
}

template <>
void caffe_cpu_axpby<float>(const long N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const long N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_axpy<size_t>(const long N, const size_t alpha, const size_t* X,
    size_t* Y) { }

template <>
void caffe_add<float>(const long n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const long n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const long n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const long n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const long n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const long n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const long n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const long n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const long n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const long n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const long n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const long n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const long n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const long n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const long n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const long n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const long n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const long n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
#ifdef DETERMINISTIC
    return 5153;
#else
    return (*caffe_rng())();
#endif
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const long n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (long i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const long n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const long n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const long n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (long i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const long n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const long n, const double mu,
                                const double sigma, double* r);

#ifdef USE_MKL
static void bernoulli_generate(long n, double p, int* r) {
  int seed = 17 + caffe_rng_rand() % 4096;

#ifdef _OPENMP
  int nthr = omp_get_max_threads();
  int threshold = nthr * caffe::cpu::OpenMpManager::getProcessorSpeedMHz() / 3;
  bool run_parallel =
    (Caffe::mode() != Caffe::GPU) &&
    (omp_in_parallel() == 0) &&
    (n >= threshold);
  if (!run_parallel) nthr = 1;

# pragma omp parallel num_threads(nthr)
  {
    const int ithr = omp_get_thread_num();
    const long avg_amount = (n + nthr - 1) / nthr;
    const long my_offset = ithr * avg_amount;
    const long my_amount = std::min(my_offset + avg_amount, n) - my_offset;
#else
  {
    const long my_amount = n;
    const long my_offset = 0;
#endif

    if (my_amount > 0) {
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, seed);
      vslSkipAheadStream(stream, my_offset);
      viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount,
        r + my_offset, p);
      vslDeleteStream(&stream);
    }
  }
}
#endif

template <typename Dtype>
void caffe_rng_bernoulli(const long n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
#ifdef USE_MKL
  bernoulli_generate(n, p, r);
#else
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (long i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
#endif
}

template
void caffe_rng_bernoulli<double>(const long n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const long n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const long n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
#ifdef USE_MKL
  bernoulli_generate(n, p, reinterpret_cast<int *>(r));
#else
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (long i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
#endif
}

template
void caffe_rng_bernoulli<double>(const long n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const long n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const long n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const long n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <>
size_t caffe_cpu_strided_dot<size_t>(const long n, const size_t* x,
        const int incx, const size_t* y, const int incy) {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype caffe_cpu_dot(const long n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const long n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const long n, const double* x, const double* y);

template
size_t caffe_cpu_dot<size_t>(const long n, const size_t* x, const size_t* y);

template <>
float caffe_cpu_asum<float>(const long n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const long n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
size_t caffe_cpu_asum<size_t>(const long n, const size_t* x) {
  NOT_IMPLEMENTED;
  return 0;
}

template <>
void caffe_cpu_scale<float>(const long n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const long n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

}  // namespace caffe
