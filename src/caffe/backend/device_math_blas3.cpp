#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device.hpp"

#ifdef USE_LIBDNN  // USE_LIBDNN
#include "caffe/libdnn/libdnn_blas.hpp"
#endif  // USE_LIBDNN

namespace caffe {

#ifdef USE_HALF
void Device::gemm_half(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const half_fp alpha,
                            vptr<const half_fp> a,
                            vptr<const half_fp> b,
                            const half_fp beta,
                            vptr<half_fp> c) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<half_fp, half_fp>()->gemm(trans_a, trans_b,
                               m, n, k, alpha,
                               a, b, beta, c,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<half_fp, half_fp> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_HALF

#ifdef USE_SINGLE
void Device::gemm_float(const CBLAS_TRANSPOSE trans_a,
                           const CBLAS_TRANSPOSE trans_b,
                           const uint_tp m, const uint_tp n, const uint_tp k,
                           const float alpha, vptr<const float> a,
                           vptr<const float> b, const float beta,
                           vptr<float> c) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<float, float>()->gemm(trans_a, trans_b,
                               m, n, k, alpha,
                               a, b, beta, c,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<float, float> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
void Device::gemm_double(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const double alpha, vptr<const double> a,
                            vptr<const double> b,
                            const double beta, vptr<double> c) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<double, double>()->gemm(trans_a, trans_b,
                               m, n, k, alpha,
                               a, b, beta, c,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<double, double> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_DOUBLE

#ifdef USE_INT_QUANT_8
void Device::gemm_int8(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const int8_t alpha, vptr<const int8_t> a,
                            vptr<const int8_t> b,
                            const int8_t beta, vptr<int8_t> c) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int8_t, int8_t>()->gemm(trans_a, trans_b,
                               m, n, k, alpha,
                               a, b, beta, c,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int8_t, int8_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_8

#ifdef USE_INT_QUANT_16
void Device::gemm_int16(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const int16_t alpha, vptr<const int16_t> a,
                            vptr<const int16_t> b,
                            const int16_t beta, vptr<int16_t> c) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int16_t, int16_t>()->gemm(trans_a, trans_b,
                               m, n, k, alpha,
                               a, b, beta, c,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int16_t, int16_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_16

#ifdef USE_INT_QUANT_32
void Device::gemm_int32(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const int32_t alpha, vptr<const int32_t> a,
                            vptr<const int32_t> b,
                            const int32_t beta, vptr<int32_t> c) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int32_t, int32_t>()->gemm(trans_a, trans_b,
                               m, n, k, alpha,
                               a, b, beta, c,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int32_t, int32_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_32

#ifdef USE_INT_QUANT_64
void Device::gemm_int64(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const int64_t alpha, vptr<const int64_t> a,
                            vptr<const int64_t> b,
                            const int64_t beta, vptr<int64_t> c) {
#ifdef USE_LIBDNN
  this->template GetLibDNNBlas<int64_t, int64_t>()->gemm(trans_a, trans_b,
                               m, n, k, alpha,
                               a, b, beta, c,
                               LIBDNN_ACCUMULATE_PREC_NATIVE,
                               make_shared<Quantizer<int64_t, int64_t> >(this));
#else  // USE_LIBDNN
  NOT_IMPLEMENTED;
#endif  // USE_LIBDNN
}
#endif  // USE_INT_QUANT_64

}  // namespace caffe
