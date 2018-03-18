#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device.hpp"

#ifdef USE_LIBDNN  // USE_LIBDNN
#include "caffe/libdnn/libdnn_blas.hpp"
#endif  // USE_LIBDNN

namespace caffe {

#ifdef USE_HALF
void Device::gemm_half(const CBLAS_TRANSPOSE trans_a,
                       const CBLAS_TRANSPOSE trans_b, const uint_tp M,
                       const uint_tp N, const uint_tp K, const half_fp alpha,
                       vptr<const half_fp> a, vptr<const half_fp> b,
                       const half_fp beta, vptr<half_fp> c,
                       const QuantizerValues* const alpha_quant,
                       const QuantizerValues* const a_quant,
                       const QuantizerValues* const b_quant,
                       const QuantizerValues* const beta_quant,
                       const QuantizerValues* const c_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->gemm(trans_a, trans_b,
                               M, N, K, alpha,
                               a, b, beta, c,
                               alpha_quant, a_quant, b_quant,
                               beta_quant, c_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_HALF

#ifdef USE_SINGLE
void Device::gemm_float(const CBLAS_TRANSPOSE trans_a,
                           const CBLAS_TRANSPOSE trans_b,
                           const uint_tp M, const uint_tp N, const uint_tp K,
                           const float alpha, vptr<const float> a,
                           vptr<const float> b, const float beta,
                           vptr<float> c,
                           const QuantizerValues* const alpha_quant,
                           const QuantizerValues* const a_quant,
                           const QuantizerValues* const b_quant,
                           const QuantizerValues* const beta_quant,
                           const QuantizerValues* const c_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->gemm(trans_a, trans_b,
                               M, N, K, alpha,
                               a, b, beta, c,
                               alpha_quant, a_quant, b_quant,
                               beta_quant, c_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
void Device::gemm_double(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const double alpha, vptr<const double> a,
                            vptr<const double> b,
                            const double beta, vptr<double> c,
                            const QuantizerValues* const alpha_quant,
                            const QuantizerValues* const a_quant,
                            const QuantizerValues* const b_quant,
                            const QuantizerValues* const beta_quant,
                            const QuantizerValues* const c_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->gemm(trans_a, trans_b,
                               M, N, K, alpha,
                               a, b, beta, c,
                               alpha_quant, a_quant, b_quant,
                               beta_quant, c_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_DOUBLE

#ifdef USE_INT_QUANT_8
void Device::gemm_uint8(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const uint8_t alpha, vptr<const uint8_t> a,
                            vptr<const uint8_t> b,
                            const uint8_t beta, vptr<uint8_t> c,
                            const QuantizerValues* const alpha_quant,
                            const QuantizerValues* const a_quant,
                            const QuantizerValues* const b_quant,
                            const QuantizerValues* const beta_quant,
                            const QuantizerValues* const c_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint8_t, uint8_t>()->gemm(trans_a, trans_b,
                               M, N, K, alpha,
                               a, b, beta, c,
                               alpha_quant, a_quant, b_quant,
                               beta_quant, c_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_8

#ifdef USE_INT_QUANT_16
void Device::gemm_uint16(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const uint16_t alpha, vptr<const uint16_t> a,
                            vptr<const uint16_t> b,
                            const uint16_t beta, vptr<uint16_t> c,
                            const QuantizerValues* const alpha_quant,
                            const QuantizerValues* const a_quant,
                            const QuantizerValues* const b_quant,
                            const QuantizerValues* const beta_quant,
                            const QuantizerValues* const c_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint16_t, uint16_t>()->gemm(trans_a, trans_b,
                               M, N, K, alpha,
                               a, b, beta, c,
                               alpha_quant, a_quant, b_quant,
                               beta_quant, c_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_16

#ifdef USE_INT_QUANT_32
void Device::gemm_uint32(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const uint32_t alpha, vptr<const uint32_t> a,
                            vptr<const uint32_t> b,
                            const uint32_t beta, vptr<uint32_t> c,
                            const QuantizerValues* const alpha_quant,
                            const QuantizerValues* const a_quant,
                            const QuantizerValues* const b_quant,
                            const QuantizerValues* const beta_quant,
                            const QuantizerValues* const c_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint32_t, uint32_t>()->gemm(trans_a, trans_b,
                               M, N, K, alpha,
                               a, b, beta, c,
                               alpha_quant, a_quant, b_quant,
                               beta_quant, c_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_32

#ifdef USE_INT_QUANT_64
void Device::gemm_uint64(const CBLAS_TRANSPOSE trans_a,
                         const CBLAS_TRANSPOSE trans_b,
                         const uint_tp M, const uint_tp N, const uint_tp K,
                         const uint64_t alpha, vptr<const uint64_t> a,
                         vptr<const uint64_t> b,
                         const uint64_t beta, vptr<uint64_t> c,
                         const QuantizerValues* const alpha_quant,
                         const QuantizerValues* const a_quant,
                         const QuantizerValues* const b_quant,
                         const QuantizerValues* const beta_quant,
                         const QuantizerValues* const c_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint64_t, uint64_t>()->gemm(trans_a, trans_b,
                               M, N, K, alpha,
                               a, b, beta, c,
                               alpha_quant, a_quant, b_quant,
                               beta_quant, c_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_64

}  // namespace caffe
