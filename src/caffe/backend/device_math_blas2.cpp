#include "caffe/backend/vptr.hpp"
#include "caffe/backend/device.hpp"

namespace caffe {

void Device::gemv_half(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                          const uint_tp n, const half_fp alpha,
                          vptr<const half_fp> a,
                          vptr<const half_fp> x,
                          const half_fp beta,
                          vptr<half_fp> y) {
  NOT_IMPLEMENTED;
}

void Device::gemv_float(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                        const uint_tp n, const float alpha,
                        vptr<const float> a,
                        vptr<const float> x, const float beta,
                        vptr<float> y) {
  NOT_IMPLEMENTED;
}

void Device::gemv_double(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                         const uint_tp n, const double alpha,
                         vptr<const double> a,
                         vptr<const double> x, const double beta,
                         vptr<double> y) {
  NOT_IMPLEMENTED;
}

}  // namespace caffe
