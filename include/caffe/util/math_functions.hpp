#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <cstring>  // for memset

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

// Forward declare
class QuantizerValues;

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template<typename Dtype>
typename std::enable_if<unsigned_integer_is_same<Dtype>::value, void>::type
caffe_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
           const int_tp M, const int_tp N, const int_tp K,
           const Dtype alpha, const Dtype* A, const Dtype* B,
           const Dtype beta, Dtype* C,
           const QuantizerValues* const alpha_quant = nullptr,
           const QuantizerValues* const a_quant = nullptr,
           const QuantizerValues* const b_quant = nullptr,
           const QuantizerValues* const beta_quant = nullptr,
           const QuantizerValues* const c_quant = nullptr);

template<typename Dtype>
typename std::enable_if<float_is_same<Dtype>::value, void>::type
caffe_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
           const int_tp M, const int_tp N, const int_tp K,
           const Dtype alpha, const Dtype* A, const Dtype* B,
           const Dtype beta, Dtype* C,
           const QuantizerValues* const alpha_quant = nullptr,
           const QuantizerValues* const a_quant = nullptr,
           const QuantizerValues* const b_quant = nullptr,
           const QuantizerValues* const beta_quant = nullptr,
           const QuantizerValues* const c_quant = nullptr);

template<typename Dtype>
typename std::enable_if<unsigned_integer_is_same<Dtype>::value, void>::type
caffe_gemv(const CBLAS_TRANSPOSE trans_A, const int_tp M,
           const int_tp N, const Dtype alpha, const Dtype* A,
           const Dtype* x, const Dtype beta, Dtype* y,
           const QuantizerValues* const alpha_quant = nullptr,
           const QuantizerValues* const a_quant = nullptr,
           const QuantizerValues* const x_quant = nullptr,
           const QuantizerValues* const beta_quant = nullptr,
           const QuantizerValues* const y_quant = nullptr);

template<typename Dtype>
typename std::enable_if<float_is_same<Dtype>::value, void>::type
caffe_gemv(const CBLAS_TRANSPOSE trans_A, const int_tp M,
           const int_tp N, const Dtype alpha, const Dtype* A,
           const Dtype* x, const Dtype beta, Dtype* y,
           const QuantizerValues* const alpha_quant = nullptr,
           const QuantizerValues* const a_quant = nullptr,
           const QuantizerValues* const x_quant = nullptr,
           const QuantizerValues* const beta_quant = nullptr,
           const QuantizerValues* const y_quant = nullptr);

template<typename Dtype>
void caffe_axpy(const int_tp N, const Dtype alpha, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_axpby(const int_tp N, const Dtype alpha, const Dtype* x,
                 const Dtype beta, Dtype* y);

template<typename Dtype>
void caffe_copy(const int_tp n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_set(const int_tp n, const Dtype alpha, Dtype *X);

inline void caffe_memset(const uint_tp n, const int_tp alpha, void* X) {
  memset(X, alpha, n);  // NOLINT(caffe/alt_fn)
}

template<typename Dtype>
void caffe_add_scalar(const int_tp n, const Dtype alpha, Dtype *x);

template<typename Dtype>
void caffe_scal(const int_tp n, const Dtype alpha, Dtype *x);

template<typename Dtype>
void caffe_sqr(const int_tp n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_sqrt(const int_tp n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_add(const int_tp n, const Dtype* a, const Dtype* b, Dtype* y);

template<typename Dtype>
void caffe_sub(const int_tp n, const Dtype* a, const Dtype* b, Dtype* y);

template<typename Dtype>
void caffe_mul(const int_tp n, const Dtype* a, const Dtype* b, Dtype* y);

template<typename Dtype>
void caffe_div(const int_tp n, const Dtype* a, const Dtype* b, Dtype* y);

template<typename Dtype>
void caffe_powx(const int_tp n, const Dtype* a, const Dtype b, Dtype* y);

uint_tp caffe_rng_rand();

template<typename Dtype>
Dtype caffe_nextafter(const Dtype b);

template<typename Dtype>
void caffe_rng_uniform(const uint_tp n, Dtype* r);

template<typename Dtype>
void caffe_rng_uniform(const int_tp n, const Dtype a, const Dtype b, Dtype* r);

template<typename Dtype>
void caffe_rng_gaussian(const int_tp n, const Dtype mu, const Dtype sigma,
                        Dtype* r);

template<typename Dtype, typename Itype>
void caffe_rng_bernoulli(const int_tp n, const Dtype p, Itype* r);

template<typename Dtype>
void caffe_exp(const int_tp n, const Dtype* a, Dtype* Y);

template<typename Dtype>
void caffe_log(const int_tp n, const Dtype* a, Dtype* Y);

template<typename Dtype>
void caffe_abs(const int_tp n, const Dtype* a, Dtype* Y);

template<typename Dtype>
Dtype caffe_dot(const int_tp n, const Dtype* x, const Dtype* y);


template<typename Dtype>
Dtype caffe_strided_dot(const int_tp n, const Dtype* x, const int_tp incx,
                        const Dtype* y, const int_tp incy);

// Returns the sum of the absolute values of the elements of vector x
template<typename Dtype>
Dtype caffe_asum(const int_tp n, const Dtype* x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template<typename Dtype>
inline int8_t caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void caffe_##name(const int_tp n, const Dtype* X, Dtype* Y) { \
    CHECK_GT(n, 0); CHECK(X); CHECK(Y); \
    for (int_tp i = 0; i < n; ++i) { \
      operation; \
    } \
  }

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, Y[i] = caffe_sign<Dtype>(X[i]))

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
    Y[i] = static_cast<bool>((std::signbit)(X[i])))

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, Y[i] = std::fabs(X[i]))

template<typename Dtype>
void caffe_scale(const int_tp n, const Dtype alpha, const Dtype *X,
                     Dtype* Y);

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
