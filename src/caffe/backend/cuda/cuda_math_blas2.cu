#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>

#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/cuda/caffe_cuda.hpp"
#include "caffe/backend/cuda/cuda_device.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"

#ifdef USE_CUDA

#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#endif  // USE_CUDA

namespace caffe {

#ifdef USE_CUDA

void cuda_device::gemv_half
              (const CBLAS_TRANSPOSE TransA, const uint_tp M,
               const uint_tp N, const half_float::half alpha,
               vptr<half_float::half> A,
               vptr<half_float::half> x, const half_float::half beta,
               vptr<half_float::half> y) {
#ifdef USE_GPU_HALF
  NOT_IMPLEMENTED;  // TODO
#else  // USE_GPU_HALF
  NOT_IMPLEMENTED;
#endif  // USE_GPU_HALF
}

void cuda_device::gemv_float
              (const CBLAS_TRANSPOSE TransA, const uint_tp M,
               const uint_tp N, const float alpha,
               vptr<float> A,
               vptr<float> x, const float beta,
               vptr<float> y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA,
                           N, M, &alpha, A.get_cuda_ptr(),
                           N, x.get_cuda_ptr(), 1, &beta, y.get_cuda_ptr(), 1));
}

void cuda_device::gemv_double
              (const CBLAS_TRANSPOSE TransA, const uint_tp M,
               const uint_tp N, const double alpha,
               vptr<double> A,
               vptr<double> x, const double beta,
               vptr<double> y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA,
                           N, M, &alpha, A.get_cuda_ptr(),
                           N, x.get_cuda_ptr(), 1, &beta, y.get_cuda_ptr(), 1));
}


#endif  // USE_CUDA

}
