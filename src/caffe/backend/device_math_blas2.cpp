#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device.hpp"

#ifdef USE_LIBDNN  // USE_LIBDNN
#include "caffe/libdnn/libdnn_blas.hpp"
#endif  // USE_LIBDNN

namespace caffe {

#ifdef USE_HALF
void Device::gemv_half(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                          const uint_tp n, const half_fp alpha,
                          vptr<const half_fp> a,
                          vptr<const half_fp> x,
                          const half_fp beta,
                          vptr<half_fp> y,
                          const QuantizerValues* const alpha_quant,
                          const QuantizerValues* const a_quant,
                          const QuantizerValues* const x_quant,
                          const QuantizerValues* const beta_quant,
                          const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->gemv(trans_a, m, n, alpha,
                               a, x, beta, y, alpha_quant, a_quant, x_quant,
                               beta_quant, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_HALF

#ifdef USE_SINGLE
void Device::gemv_float(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                        const uint_tp n, const float alpha,
                        vptr<const float> a,
                        vptr<const float> x, const float beta,
                        vptr<float> y,
                        const QuantizerValues* const alpha_quant,
                        const QuantizerValues* const a_quant,
                        const QuantizerValues* const x_quant,
                        const QuantizerValues* const beta_quant,
                        const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->gemv(trans_a, m, n, alpha,
                               a, x, beta, y, alpha_quant, a_quant, x_quant,
                               beta_quant, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
void Device::gemv_double(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                         const uint_tp n, const double alpha,
                         vptr<const double> a,
                         vptr<const double> x, const double beta,
                         vptr<double> y,
                         const QuantizerValues* const alpha_quant,
                         const QuantizerValues* const a_quant,
                         const QuantizerValues* const x_quant,
                         const QuantizerValues* const beta_quant,
                         const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->gemv(trans_a, m, n, alpha,
                               a, x, beta, y, alpha_quant, a_quant, a_quant,
                               beta_quant, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_DOUBLE

#ifdef USE_INT_QUANT_8
void Device::gemv_uint8(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                          const uint_tp n, const uint8_t alpha,
                          vptr<const uint8_t> a,
                          vptr<const uint8_t> x,
                          const uint8_t beta,
                          vptr<uint8_t> y,
                          const QuantizerValues* const alpha_quant,
                          const QuantizerValues* const a_quant,
                          const QuantizerValues* const x_quant,
                          const QuantizerValues* const beta_quant,
                          const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint8_t, uint8_t>()->gemv(trans_a, m, n, alpha,
                               a, x, beta, y, alpha_quant, a_quant,
                               x_quant, beta_quant, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_8

#ifdef USE_INT_QUANT_16
void Device::gemv_uint16(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                         const uint_tp n, const uint16_t alpha,
                         vptr<const uint16_t> a,
                         vptr<const uint16_t> x,
                         const uint16_t beta,
                         vptr<uint16_t> y,
                         const QuantizerValues* const alpha_quant,
                         const QuantizerValues* const a_quant,
                         const QuantizerValues* const x_quant,
                         const QuantizerValues* const beta_quant,
                         const QuantizerValues* const y_quant) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint16_t, uint16_t>()->gemv(trans_a, m, n, alpha,
                               a, x, beta, y, alpha_quant, a_quant, x_quant,
                               beta_quant, y_quant);
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_16

#ifdef USE_INT_QUANT_32
void Device::gemv_uint32(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                          const uint_tp n, const uint32_t alpha,
                          vptr<const uint32_t> a,
                          vptr<const uint32_t> x,
                          const uint32_t beta,
                          vptr<uint32_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint32_t, uint32_t>()->gemv(trans_a, m, n, alpha,
                               a, x, beta, y,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<uint32_t, uint32_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_32

#ifdef USE_INT_QUANT_64
void Device::gemv_uint64(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                          const uint_tp n, const uint64_t alpha,
                          vptr<const uint64_t> a,
                          vptr<const uint64_t> x,
                          const uint64_t beta,
                          vptr<uint64_t> y) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<uint64_t, uint64_t>()->gemv(trans_a, m, n, alpha,
                               a, x, beta, y,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<uint64_t, uint64_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_64

}  // namespace caffe
