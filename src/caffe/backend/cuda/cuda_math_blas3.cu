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

void CudaDevice::gemm_half(const CBLAS_TRANSPOSE trans_a,
                           const CBLAS_TRANSPOSE trans_b,
                           const uint_tp m, const uint_tp n, const uint_tp k,
                           const half_fp alpha,
                           vptr<const half_fp> a,
                           vptr<const half_fp> b,
                           const half_fp beta,
                           vptr<half_fp> c) {
  // Note that cublas follows fortran order.
  int_tp lda = (trans_a == CblasNoTrans) ? k : m;
  int_tp ldb = (trans_b == CblasNoTrans) ? n : k;
  cublasOperation_t cuTransA =
      (trans_a == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (trans_b == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasHgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
                      n, m, k, reinterpret_cast<const half*>(&alpha),
                      reinterpret_cast<const half*>(b.get_cuda_ptr()),
                      ldb, reinterpret_cast<const half*>(a.get_cuda_ptr()),
                      lda, reinterpret_cast<const half*>(&beta),
                      reinterpret_cast<half*>(c.get_cuda_ptr()), n));
}

void CudaDevice::gemm_float(const CBLAS_TRANSPOSE trans_a,
                            const CBLAS_TRANSPOSE trans_b,
                            const uint_tp m, const uint_tp n, const uint_tp k,
                            const float alpha, vptr<const float> a,
                            vptr<const float> b, const float beta,
                            vptr<float> c) {
  // Note that cublas follows fortran order.
  int_tp lda = (trans_a == CblasNoTrans) ? k : m;
  int_tp ldb = (trans_b == CblasNoTrans) ? n : k;
  cublasOperation_t cuTransA =
      (trans_a == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (trans_b == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
                      n, m, k, &alpha, b.get_cuda_ptr(),
                      ldb, a.get_cuda_ptr(), lda, &beta, c.get_cuda_ptr(), n));
}

void CudaDevice::gemm_double(const CBLAS_TRANSPOSE trans_a,
                             const CBLAS_TRANSPOSE trans_b,
                             const uint_tp m, const uint_tp n, const uint_tp k,
                             const double alpha, vptr<const double> a,
                             vptr<const double> b, const double beta,
                             vptr<double> c) {
  // Note that cublas follows fortran order.
  int_tp lda = (trans_a == CblasNoTrans) ? k : m;
  int_tp ldb = (trans_b == CblasNoTrans) ? n : k;
  cublasOperation_t cuTransA =
      (trans_a == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (trans_b == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
                           n, m, k, &alpha, b.get_cuda_ptr(),
                           ldb, a.get_cuda_ptr(), lda, &beta,
                           c.get_cuda_ptr(), n));
}


#endif  // USE_CUDA


}
