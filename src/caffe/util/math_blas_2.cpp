#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

// Integer quantized types
template<typename Dtype>
typename std::enable_if<unsigned_integer_is_same<Dtype>::value, void>::type
caffe_gemv(const CBLAS_TRANSPOSE trans_A,
           const int_tp M, const int_tp N, const Dtype alpha,
           const Dtype* a, const Dtype* x,
           const Dtype beta, Dtype* y,
           const QuantizerValues* const alpha_quant,
           const QuantizerValues* const a_quant,
           const QuantizerValues* const x_quant,
           const QuantizerValues* const beta_quant,
           const QuantizerValues* const y_quant) {

  typedef typename std::conditional<sizeof(Dtype) == 1, int16_t,
          typename std::conditional<sizeof(Dtype) == 2, int32_t,
                                    int64_t>::type>::type Difftype;
  typedef typename std::conditional<sizeof(Dtype) == 1,
                                    int32_t, int64_t>::type Acctype;

  int8_t shift_bits = (32/sizeof(Dtype)) - 1;

  int32_t mult;
  int8_t shift;
  int32_t alpha_mult;
  int8_t alpha_shift;
  int32_t beta_mult;
  int8_t beta_shift;
  Acctype y_max = y_quant->get_max<Acctype>();
  Acctype y_min = y_quant->get_min<Acctype>();
  Dtype lhs_off = a_quant->get_zero<Dtype>();
  Dtype rhs_off = x_quant->get_zero<Dtype>();
  Dtype alpha_off = alpha_quant ? alpha_quant->get_zero<Dtype>() : Dtype(0);
  Dtype beta_off = beta_quant ? beta_quant->get_zero<Dtype>() : Dtype(0);
  const Acctype result_off = y_quant->get_zero<Acctype>();

  QuantizerBase::template MultiplicativeQuantVals<int32_t>(
      a_quant, x_quant, y_quant, &mult, &shift, shift_bits);
  QuantizerBase::template MultiplicativeQuantVals<int32_t>(
      y_quant, alpha_quant, y_quant, &alpha_mult, &alpha_shift, shift_bits);
  QuantizerBase::template MultiplicativeQuantVals<int32_t>(
      y_quant, beta_quant, y_quant, &beta_mult, &beta_shift, shift_bits);

  int_tp a_inc = (trans_A == CblasNoTrans) ? 1 : N;
  int_tp y_cnt = (trans_A == CblasNoTrans) ? M : N;
  int_tp x_cnt = (trans_A == CblasNoTrans) ? N : M;
#pragma omp parallel for
  for (int_tp m = 0; m < y_cnt; m++) {
    int_tp a_index = (trans_A == CblasNoTrans) ? m * N : m;
    Acctype acc = 0;
    for (int_tp n = 0; n < x_cnt; n++) {
      Difftype a_diff = a[a_index] - lhs_off;
      Difftype x_diff = x[n] - rhs_off;
      acc += static_cast<Acctype>(a_diff) * static_cast<Acctype>(x_diff);
      a_index += a_inc;
    }
    Acctype reg = acc * (alpha_quant ? Acctype(1) : alpha);
    reg = static_cast<Acctype>((static_cast<int64_t>(reg) *
                           static_cast<int64_t>(mult)) / (1ll << shift_bits));
    if (shift >= 0) {
      reg = reg >> shift;
    } else {
      reg = reg << -shift;
    }
    if (alpha_quant) {
      Difftype alpha_diff = alpha - alpha_off;
      reg = static_cast<Acctype>(alpha_diff) * static_cast<Acctype>(reg);
      reg = static_cast<Acctype>((static_cast<int64_t>(reg) *
                     static_cast<int64_t>(alpha_mult)) / (1ll << shift_bits));
      if (alpha_shift >= 0) {
        reg = reg >> alpha_shift;
      } else {
        reg = reg << -alpha_shift;
      }
    }
    if (beta_quant) {
      Difftype beta_diff = beta - beta_off;
      Difftype c_diff = y[m] - static_cast<Difftype>(result_off);
      Acctype creg = static_cast<Acctype>(beta_diff)
                   * static_cast<Acctype>(c_diff);
      creg = static_cast<Acctype>((static_cast<int64_t>(creg) *
                      static_cast<int64_t>(beta_mult)) / (1ll << shift_bits));
      if (beta_shift >= 0) {
        creg = creg >> beta_shift;
      } else {
        creg = creg << -beta_shift;
      }
      reg = reg + creg;
    } else if (beta == Dtype(1)) {
      reg = reg + (y[m] - result_off);
    }
    reg = reg + result_off;
    y[m] = static_cast<Dtype>(std::min(std::max(reg, y_min), y_max));
  }
}

// Half precision
template<typename Dtype>
typename std::enable_if<float_is_same<Dtype>::value, void>::type
caffe_gemv(const CBLAS_TRANSPOSE trans_A,
           const int_tp M, const int_tp N, const Dtype alpha,
           const Dtype* a, const Dtype* x,
           const Dtype beta, Dtype* y,
           const QuantizerValues* const alpha_quant,
           const QuantizerValues* const a_quant,
           const QuantizerValues* const x_quant,
           const QuantizerValues* const beta_quant,
           const QuantizerValues* const y_quant) {
  int_tp a_inc = (trans_A == CblasNoTrans) ? 1 : N;
  int_tp y_cnt = (trans_A == CblasNoTrans) ? M : N;
  int_tp x_cnt = (trans_A == CblasNoTrans) ? N : M;
  for (int_tp m = 0; m < y_cnt; m++) {
    int_tp a_index = (trans_A == CblasNoTrans) ? m * N : m;
    Dtype acc = 0;
    for (int_tp n = 0; n < x_cnt; n++) {
      acc += a[a_index] * x[n];
      a_index += a_inc;
    }
    if (beta == 0)
      y[m] = acc * alpha;
    else
      y[m] = acc * alpha + beta * y[m];
  }
}

template
typename std::enable_if<float_is_same<half_fp>::value, void>::type
caffe_gemv<half_fp>(const CBLAS_TRANSPOSE trans_A,
                    const int_tp M, const int_tp N,
                    const half_fp alpha,
                    const half_fp* a, const half_fp* x,
                    const half_fp beta, half_fp* y,
                    const QuantizerValues* const alpha_quant,
                    const QuantizerValues* const a_quant,
                    const QuantizerValues* const x_quant,
                    const QuantizerValues* const beta_quant,
                    const QuantizerValues* const y_quant);
template
typename std::enable_if<unsigned_integer_is_same<uint8_t>::value, void>::type
caffe_gemv<uint8_t>(const CBLAS_TRANSPOSE trans_A,
                    const int_tp M, const int_tp N,
                    const uint8_t alpha,
                    const uint8_t* a, const uint8_t* x,
                    const uint8_t beta, uint8_t* y,
                    const QuantizerValues* const alpha_quant,
                    const QuantizerValues* const a_quant,
                    const QuantizerValues* const x_quant,
                    const QuantizerValues* const beta_quant,
                    const QuantizerValues* const y_quant);
template
typename std::enable_if<unsigned_integer_is_same<uint16_t>::value, void>::type
caffe_gemv<uint16_t>(const CBLAS_TRANSPOSE trans_A,
                     const int_tp M, const int_tp N,
                     const uint16_t alpha,
                     const uint16_t* a, const uint16_t* x,
                     const uint16_t beta, uint16_t* y,
                     const QuantizerValues* const alpha_quant,
                     const QuantizerValues* const a_quant,
                     const QuantizerValues* const x_quant,
                     const QuantizerValues* const beta_quant,
                     const QuantizerValues* const y_quant);
template
typename std::enable_if<unsigned_integer_is_same<uint32_t>::value, void>::type
caffe_gemv<uint32_t>(const CBLAS_TRANSPOSE trans_A,
                     const int_tp M, const int_tp N,
                     const uint32_t alpha,
                     const uint32_t* a, const uint32_t* x,
                     const uint32_t beta, uint32_t* y,
                     const QuantizerValues* const alpha_quant,
                     const QuantizerValues* const a_quant,
                     const QuantizerValues* const x_quant,
                     const QuantizerValues* const beta_quant,
                     const QuantizerValues* const y_quant);

template
typename std::enable_if<unsigned_integer_is_same<uint64_t>::value, void>::type
caffe_gemv<uint64_t>(const CBLAS_TRANSPOSE trans_A,
                     const int_tp M, const int_tp N,
                     const uint64_t alpha,
                     const uint64_t* a, const uint64_t* x,
                     const uint64_t beta, uint64_t* y,
                     const QuantizerValues* const alpha_quant,
                     const QuantizerValues* const a_quant,
                     const QuantizerValues* const x_quant,
                     const QuantizerValues* const beta_quant,
                     const QuantizerValues* const y_quant);

template<>
void caffe_gemv<float>(const CBLAS_TRANSPOSE trans_A, const int_tp M,
                       const int_tp N, const float alpha, const float* A,
                       const float* x, const float beta, float* y,
                       const QuantizerValues* const alpha_quant,
                       const QuantizerValues* const a_quant,
                       const QuantizerValues* const x_quant,
                       const QuantizerValues* const beta_quant,
                       const QuantizerValues* const y_quant) {
  cblas_sgemv(CblasRowMajor, trans_A, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template<>
void caffe_gemv<double>(const CBLAS_TRANSPOSE trans_A, const int_tp M,
                        const int_tp N, const double alpha, const double* A,
                        const double* x, const double beta, double* y,
                        const QuantizerValues* const alpha_quant,
                        const QuantizerValues* const a_quant,
                        const QuantizerValues* const x_quant,
                        const QuantizerValues* const beta_quant,
                        const QuantizerValues* const y_quant) {
  cblas_dgemv(CblasRowMajor, trans_A, M, N, alpha, A, N, x, 1, beta, y, 1);
}

}  // namespace caffe

