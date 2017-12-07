#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>

#include "caffe/backend/cuda/cuda_device.hpp"
#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/cuda/caffe_cuda.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"

#ifdef USE_CUDA
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#endif  // USE_CUDA

namespace caffe {

#ifdef USE_CUDA

void CudaDevice::gemv_half(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                           const uint_tp n, const half_fp alpha,
                           vptr<const half_fp> a,
                           vptr<const half_fp> x,
                           const half_fp beta,
                           vptr<half_fp> y) {
#ifdef USE_HALF
  NOT_IMPLEMENTED;  // TODO
#else  // USE_HALF
  NOT_IMPLEMENTED;
#endif  // USE_HALF
}

void CudaDevice::gemv_float(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                            const uint_tp n, const float alpha,
                            vptr<const float> a,
                            vptr<const float> x, const float beta,
                            vptr<float> y) {
  cublasOperation_t cuTransA =
      (trans_a == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA,
                           n, m, &alpha, a.get_cuda_ptr(),
                           n, x.get_cuda_ptr(), 1, &beta, y.get_cuda_ptr(), 1));
}

void CudaDevice::gemv_double(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                             const uint_tp n, const double alpha,
                             vptr<const double> a,
                             vptr<const double> x, const double beta,
                             vptr<double> y) {
  cublasOperation_t cuTransA =
      (trans_a == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA,
                           n, m, &alpha, a.get_cuda_ptr(),
                           n, x.get_cuda_ptr(), 1, &beta, y.get_cuda_ptr(), 1));
}


#endif  // USE_CUDA

}
