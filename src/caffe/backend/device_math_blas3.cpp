#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device.hpp"

namespace caffe {

void Device::gemm_half(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const half_fp alpha,
                            vptr<const half_fp> a,
                            vptr<const half_fp> b,
                            const half_fp beta,
                            vptr<half_fp> c) {
  NOT_IMPLEMENTED;
}

void Device::gemm_float(const CBLAS_TRANSPOSE trans_a,
                           const CBLAS_TRANSPOSE trans_b,
                           const uint_tp m, const uint_tp n, const uint_tp k,
                           const float alpha, vptr<const float> a,
                           vptr<const float> b, const float beta,
                           vptr<float> c) {
  NOT_IMPLEMENTED;
}

void Device::gemm_double(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const double alpha, vptr<const double> a,
                            vptr<const double> b,
                            const double beta, vptr<double> c) {
  NOT_IMPLEMENTED;
}

}  // namespace caffe
