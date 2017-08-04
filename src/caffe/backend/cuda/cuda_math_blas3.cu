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

void cuda_device::gemm_half(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const half_float::half alpha,
                            vptr<half_float::half> A,
                            vptr<half_float::half> B,
                            const half_float::half beta,
                            vptr<half_float::half> C) {
  // Note that cublas follows fortran order.
  int_tp lda = (TransA == CblasNoTrans) ? K : M;
  int_tp ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasHgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
                      N, M, K, static_cast<half*>(&alpha),
                      static_cast<half*>(B.get_cuda_ptr()),
                      ldb, static_cast<half*>(A.get_cuda_ptr()),
                      lda, static_cast<half*>(&beta),
                      static_cast<half*>(C.get_cuda_ptr()), N));
}

void cuda_device::gemm_float(const CBLAS_TRANSPOSE TransA,
                             const CBLAS_TRANSPOSE TransB,
                             const uint_tp M, const uint_tp N, const uint_tp K,
                             const float alpha, vptr<float> A, vptr<float> B,
                             const float beta, vptr<float> C) {
  // Note that cublas follows fortran order.
  int_tp lda = (TransA == CblasNoTrans) ? K : M;
  int_tp ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
                      N, M, K, &alpha, B.get_cuda_ptr(),
                      ldb, A.get_cuda_ptr(), lda, &beta, C.get_cuda_ptr(), N));
}

void cuda_device::gemm_double(const CBLAS_TRANSPOSE TransA,
                              const CBLAS_TRANSPOSE TransB,
                              const uint_tp M, const uint_tp N, const uint_tp K,
                              const double alpha, vptr<double> A,
                              vptr<double> B, const double beta,
                              vptr<double> C) {
  // Note that cublas follows fortran order.
  int_tp lda = (TransA == CblasNoTrans) ? K : M;
  int_tp ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
                           N, M, K, &alpha, B.get_cuda_ptr(),
                           ldb, A.get_cuda_ptr(), lda, &beta,
                           C.get_cuda_ptr(), N));
}


#endif  // USE_CUDA


}
