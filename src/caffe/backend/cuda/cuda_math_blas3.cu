#ifdef USE_CUDA
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

#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

namespace caffe {

#ifdef USE_HALF
void CudaDevice::gemm_half(const CBLAS_TRANSPOSE trans_A,
                           const CBLAS_TRANSPOSE trans_B,
                           const uint_tp M, const uint_tp N, const uint_tp K,
                           const half_fp alpha,
                           vptr<const half_fp> A,
                           vptr<const half_fp> B,
                           const half_fp beta,
                           vptr<half_fp> C,
                           const QuantizerValues* const alpha_quant,
                           const QuantizerValues* const a_quant,
                           const QuantizerValues* const b_quant,
                           const QuantizerValues* const beta_quant,
                           const QuantizerValues* const c_quant) {
  // Note that cublas follows fortran order.
  int_tp lda = (trans_A == CblasNoTrans) ? K : M;
  int_tp ldb = (trans_B == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasHgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
                      N, M, K, reinterpret_cast<const half*>(&alpha),
                      reinterpret_cast<const half*>(B.get_cuda_ptr()),
                      ldb, reinterpret_cast<const half*>(A.get_cuda_ptr()),
                      lda, reinterpret_cast<const half*>(&beta),
                      reinterpret_cast<half*>(C.get_cuda_ptr()), N));
}
#endif  // USE_HALF

#ifdef USE_SINGLE
void CudaDevice::gemm_float(const CBLAS_TRANSPOSE trans_A,
                            const CBLAS_TRANSPOSE trans_B,
                            const uint_tp M, const uint_tp N, const uint_tp K,
                            const float alpha, vptr<const float> A,
                            vptr<const float> B, const float beta,
                            vptr<float> C,
                            const QuantizerValues* const alpha_quant,
                            const QuantizerValues* const a_quant,
                            const QuantizerValues* const b_quant,
                            const QuantizerValues* const beta_quant,
                            const QuantizerValues* const c_quant) {
  // Note that cublas follows fortran order.
  int_tp lda = (trans_A == CblasNoTrans) ? K : M;
  int_tp ldb = (trans_B == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
                      N, M, K, &alpha, B.get_cuda_ptr(),
                      ldb, A.get_cuda_ptr(), lda, &beta, C.get_cuda_ptr(), N));
}
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
void CudaDevice::gemm_double(const CBLAS_TRANSPOSE trans_A,
                             const CBLAS_TRANSPOSE trans_B,
                             const uint_tp M, const uint_tp N, const uint_tp K,
                             const double alpha, vptr<const double> A,
                             vptr<const double> B, const double beta,
                             vptr<double> C,
                             const QuantizerValues* const alpha_quant,
                             const QuantizerValues* const a_quant,
                             const QuantizerValues* const b_quant,
                             const QuantizerValues* const beta_quant,
                             const QuantizerValues* const c_quant) {
  // Note that cublas follows fortran order.
  int_tp lda = (trans_A == CblasNoTrans) ? K : M;
  int_tp ldb = (trans_B == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (trans_A == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (trans_B == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
                           N, M, K, &alpha, B.get_cuda_ptr(),
                           ldb, A.get_cuda_ptr(), lda, &beta,
                           C.get_cuda_ptr(), N));
}
#endif  // USE_DOUBLE

}  // namespace caffe
#endif  // USE_CUDA
