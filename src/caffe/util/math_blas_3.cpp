#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


#ifdef USE_GEMMLOWP
#include "public/gemmlowp.h"
#endif

namespace caffe {

template<typename Dtype>
typename std::enable_if<unsigned_integer_is_same<Dtype>::value, bool>::type
gemmlowp_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
              const int_tp M, const int_tp N, const int_tp K,
              const Dtype alpha, const Dtype* A,
              const Dtype* B, const Dtype beta, Dtype* C,
              const QuantizerValues* const alpha_quant,
              const QuantizerValues* const a_quant,
              const QuantizerValues* const b_quant,
              const QuantizerValues* const beta_quant,
              const QuantizerValues* const c_quant) {
  // No gemmlowp available
  return false;
}

#ifdef USE_GEMMLOWP
template<>
typename std::enable_if<unsigned_integer_is_same<uint8_t>::value, bool>::type
gemmlowp_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
              const int_tp M, const int_tp N, const int_tp K,
              const uint8_t alpha, const uint8_t* A,
              const uint8_t* B, const uint8_t beta, uint8_t* C,
              const QuantizerValues* const alpha_quant,
              const QuantizerValues* const a_quant,
              const QuantizerValues* const b_quant,
              const QuantizerValues* const beta_quant,
              const QuantizerValues* const c_quant) {
  CHECK(a_quant && b_quant && c_quant)
       << "Integer type requires quantization values.";
  if (alpha == 1 && beta == 0 && alpha_quant == nullptr
                              && beta_quant == nullptr) {
    const int32_t lhs_offset = -a_quant->get_zero<int32_t>();
    const int32_t rhs_offset = -b_quant->get_zero<int32_t>();

    gemmlowp::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
    quantize_down_stage.result_offset_after_shift =
        c_quant->get_zero<int32_t>();

    int32_t mult;
    int8_t shift;

    QuantizerBase::template MultiplicativeQuantVals<int32_t>(
        a_quant, b_quant, c_quant, &mult, &shift);

    quantize_down_stage.result_fixedpoint_multiplier = mult;
    quantize_down_stage.result_shift = static_cast<int32_t>(shift);

    if (quantize_down_stage.result_shift < 0) {
      // Gemmlowp doesn't handle negative exponents
      return false;
    }

    gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
    const auto& output_pipeline =
        std::make_tuple(quantize_down_stage, saturating_cast_stage);

    gemmlowp::GemmContext gemm_context;

    gemmlowp::MatrixMap<uint8_t, gemmlowp::MapOrder::RowMajor> rs(C, M, N);

    if (trans_A == CblasNoTrans && trans_B == CblasNoTrans) {
      gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor>
                                                                   lhs(A, M, K);
      gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor>
                                                                   rhs(B, K, N);
      gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t,
                                       gemmlowp::DefaultL8R8BitDepthParams>(
         &gemm_context, lhs, rhs, &rs, lhs_offset, rhs_offset, output_pipeline);
    } else if (trans_A == CblasNoTrans && trans_B == CblasTrans) {
      gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor>
                                                                   lhs(A, M, K);
      gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor>
                                                                   rhs(B, K, N);
      gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t,
                                       gemmlowp::DefaultL8R8BitDepthParams>(
         &gemm_context, lhs, rhs, &rs, lhs_offset, rhs_offset, output_pipeline);
    } else if (trans_A == CblasTrans && trans_B == CblasNoTrans) {
      gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor>
                                                                   lhs(A, M, K);
      gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor>
                                                                   rhs(B, K, N);
      gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t,
                                       gemmlowp::DefaultL8R8BitDepthParams>(
         &gemm_context, lhs, rhs, &rs, lhs_offset, rhs_offset, output_pipeline);
    } else {
      gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor>
                                                                   lhs(A, M, K);
      gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor>
                                                                   rhs(B, K, N);
      gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t,
                                       gemmlowp::DefaultL8R8BitDepthParams>(
         &gemm_context, lhs, rhs, &rs, lhs_offset, rhs_offset, output_pipeline);
    }
    return true;
  }
  // Gemmlowp can't handle the case
  return false;
}
#endif  // USE_GEMMLOWP


template<typename Dtype>
typename std::enable_if<unsigned_integer_is_same<Dtype>::value, void>::type
caffe_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
           const int_tp M, const int_tp N, const int_tp K,
           const Dtype alpha, const Dtype* A,
           const Dtype* B, const Dtype beta, Dtype* C,
           const QuantizerValues* const alpha_quant,
           const QuantizerValues* const a_quant,
           const QuantizerValues* const b_quant,
           const QuantizerValues* const beta_quant,
           const QuantizerValues* const c_quant) {
  CHECK(a_quant && b_quant && c_quant)
       << "Integer type requires quantization values.";

  if (gemmlowp_gemm<Dtype>(trans_A, trans_B, M, N, K, alpha, A, B, beta, C,
                          alpha_quant, a_quant, b_quant, beta_quant, c_quant)) {
    // Handled by gemmlowp implementation
    return;
  }

  typedef typename std::conditional<sizeof(Dtype) == 1, int16_t,
          typename std::conditional<sizeof(Dtype) == 2, int32_t,
                                    int64_t>::type>::type Difftype;
  typedef typename std::conditional<sizeof(Dtype) == 1,
                                    int32_t, int64_t>::type Acctype;

  // std::cout << "Difftype: " << sizeof(Difftype) << std::endl;
  // std::cout << "Acctype: " << sizeof(Acctype) << std::endl;

  int8_t shift_bits = (32 / sizeof(Dtype)) - 1;

  int32_t mult;
  int8_t shift;
  int32_t alpha_mult;
  int8_t alpha_shift;
  int32_t beta_mult;
  int8_t beta_shift;
  Acctype c_max = c_quant->get_max<Acctype>();
  Acctype c_min = c_quant->get_min<Acctype>();
  Dtype lhs_off = a_quant->get_zero<Dtype>();
  Dtype rhs_off = b_quant->get_zero<Dtype>();
  Dtype alpha_off = alpha_quant ? alpha_quant->get_zero<Dtype>() : Dtype(0);
  Dtype beta_off = beta_quant ? beta_quant->get_zero<Dtype>() : Dtype(0);
  const Acctype result_off = c_quant->get_zero<Acctype>();

  QuantizerBase::template MultiplicativeQuantVals<int32_t>(
      a_quant, b_quant, c_quant, &mult, &shift, shift_bits);
  QuantizerBase::template MultiplicativeQuantVals<int32_t>(
      c_quant, alpha_quant, c_quant, &alpha_mult, &alpha_shift, shift_bits);
  QuantizerBase::template MultiplicativeQuantVals<int32_t>(
      c_quant, beta_quant, c_quant, &beta_mult, &beta_shift, shift_bits);

  int_tp inc_a = (trans_A == CblasNoTrans) ? 1 : M;
  int_tp inc_b = (trans_B == CblasNoTrans) ? N : 1;
  for (int_tp m = 0; m < M; ++m) {
#pragma omp parallel for
    for (int_tp n = 0; n < N; ++n) {
      Acctype acc = Acctype(0);
      int_tp b_index = trans_B == CblasNoTrans ? n : K * n;
      int_tp a_index = trans_A == CblasNoTrans ? K * m : m;
      for (int_tp k = 0; k < K; ++k) {
        Difftype a_diff = A[a_index] - lhs_off;
        Difftype b_diff = B[b_index] - rhs_off;
        // std::cout << "a_diff: " << a_diff << std::endl;
        // std::cout << "b_diff: " << b_diff << std::endl;
        acc += static_cast<Acctype>(a_diff) * static_cast<Acctype>(b_diff);
        a_index += inc_a;
        b_index += inc_b;
      }
      Acctype reg = acc * (alpha_quant ? Acctype(1) : alpha);
      // std::cout << "1: " << m << ", "<< n << ": " << reg << std::endl;
      // std::cout << "M: " << mult << std::endl;
      // std::cout << "S: " << static_cast<int32_t>(shift_bits) << std::endl;
      reg = static_cast<Acctype>((static_cast<int64_t>(reg) *
                             static_cast<int64_t>(mult)) / (1ll << shift_bits));
      // std::cout << "2: " << m << ", "<< n << ": " << reg << std::endl;
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
      // std::cout << "3: " << m << ", "<< n << ": " << reg << std::endl;
      if (beta_quant) {
        Difftype beta_diff = beta - beta_off;
        Difftype c_diff = C[m * N + n] - static_cast<Difftype>(result_off);
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
        reg = reg + (C[m * N + n] - result_off);
      }
      reg = reg + result_off;
      // std::cout << "4: " << m << ", "<< n << ": " << reg << std::endl;
      C[m * N + n] = static_cast<Dtype>(std::min(std::max(reg, c_min), c_max));
    }
  }
}

template<typename Dtype>
typename std::enable_if<float_is_same<Dtype>::value, void>::type
caffe_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
           const int_tp M, const int_tp N, const int_tp K,
           const Dtype alpha, const Dtype* A,
           const Dtype* B, const Dtype beta, Dtype* C,
           const QuantizerValues* const alpha_quant,
           const QuantizerValues* const a_quant,
           const QuantizerValues* const b_quant,
           const QuantizerValues* const beta_quant,
           const QuantizerValues* const c_quant) {
  int_tp inc_a = (trans_A == CblasNoTrans) ? 1 : M;
  int_tp inc_b = (trans_B == CblasNoTrans) ? N : 1;
  for (int_tp m = 0; m < M; ++m) {
#pragma omp parallel for
    for (int_tp n = 0; n < N; ++n) {
      Dtype acc = 0;
      int_tp b_index = trans_B == CblasNoTrans ? n : K * n;
      int_tp a_index = trans_A == CblasNoTrans ? K * m : m;
      for (int_tp k = 0; k < K; ++k) {
        acc += A[a_index] * B[b_index];
        a_index += inc_a;
        b_index += inc_b;
      }
      if (beta != 0) {
        C[m * N + n] = acc * alpha + beta * C[m * N + n];
      }
      else {
        C[m * N + n] = acc * alpha;
      }
    }
  }
}

template
typename std::enable_if<float_is_same<half_fp>::value, void>::type
caffe_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
                const int_tp M, const int_tp N, const int_tp K,
                const half_fp alpha, const half_fp* A,
                const half_fp* B, const half_fp beta, half_fp* C,
                const QuantizerValues* const alpha_quant,
                const QuantizerValues* const a_quant,
                const QuantizerValues* const b_quant,
                const QuantizerValues* const beta_quant,
                const QuantizerValues* const c_quant);
template
typename std::enable_if<unsigned_integer_is_same<uint8_t>::value, void>::type
caffe_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
                const int_tp M, const int_tp N, const int_tp K,
                const uint8_t alpha, const uint8_t* A,
                const uint8_t* B, const uint8_t beta, uint8_t* C,
                const QuantizerValues* const alpha_quant,
                const QuantizerValues* const a_quant,
                const QuantizerValues* const b_quant,
                const QuantizerValues* const beta_quant,
                const QuantizerValues* const c_quant);
template
typename std::enable_if<unsigned_integer_is_same<uint16_t>::value, void>::type
caffe_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
           const int_tp M, const int_tp N, const int_tp K,
           const uint16_t alpha, const uint16_t* A,
           const uint16_t* B, const uint16_t beta, uint16_t* C,
           const QuantizerValues* const alpha_quant,
           const QuantizerValues* const a_quant,
           const QuantizerValues* const b_quant,
           const QuantizerValues* const beta_quant,
           const QuantizerValues* const c_quant);
template
typename std::enable_if<unsigned_integer_is_same<uint32_t>::value, void>::type
caffe_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
           const int_tp M, const int_tp N, const int_tp K,
           const uint32_t alpha, const uint32_t* A,
           const uint32_t* B, const uint32_t beta, uint32_t* C,
           const QuantizerValues* const alpha_quant,
           const QuantizerValues* const a_quant,
           const QuantizerValues* const b_quant,
           const QuantizerValues* const beta_quant,
           const QuantizerValues* const c_quant);
template
typename std::enable_if<unsigned_integer_is_same<uint64_t>::value, void>::type
caffe_gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
           const int_tp M, const int_tp N, const int_tp K,
           const uint64_t alpha, const uint64_t* A,
           const uint64_t* B, const uint64_t beta, uint64_t* C,
           const QuantizerValues* const alpha_quant,
           const QuantizerValues* const a_quant,
           const QuantizerValues* const b_quant,
           const QuantizerValues* const beta_quant,
           const QuantizerValues* const c_quant);


template<>
typename std::enable_if<float_is_same<float>::value, void>::type
caffe_gemm<float>(const CBLAS_TRANSPOSE trans_A,
                  const CBLAS_TRANSPOSE trans_B, const int_tp M,
                  const int_tp N, const int_tp K, const float alpha,
                  const float* A, const float* B, const float beta,
                  float* C,
                  const QuantizerValues* const alpha_quant,
                  const QuantizerValues* const a_quant,
                  const QuantizerValues* const b_quant,
                  const QuantizerValues* const beta_quant,
                  const QuantizerValues* const c_quant) {
  int_tp lda = (trans_A == CblasNoTrans) ? K : M;
  int_tp ldb = (trans_B == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

template<>
typename std::enable_if<float_is_same<float>::value, void>::type
caffe_gemm<double>(const CBLAS_TRANSPOSE trans_A,
                   const CBLAS_TRANSPOSE trans_B, const int_tp M,
                   const int_tp N, const int_tp K, const double alpha,
                   const double* A, const double* B, const double beta,
                   double* C,
                   const QuantizerValues* const alpha_quant,
                   const QuantizerValues* const a_quant,
                   const QuantizerValues* const b_quant,
                   const QuantizerValues* const beta_quant,
                   const QuantizerValues* const c_quant) {
  int_tp lda = (trans_A == CblasNoTrans) ? K : M;
  int_tp ldb = (trans_B == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

}  // namespace caffe

