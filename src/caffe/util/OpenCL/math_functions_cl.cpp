#ifdef USE_OPENCL

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/OpenCL/OpenCLSupport.hpp"
#include "caffe/util/benchmark.hpp"
#include "cblas.h"

namespace caffe {

  void caffe_gpu_memcpy(const size_t N, const void* X, void* Y, int type) {

    if (X != Y) {
      BOOL_CHECK(caffe::OpenCL::clMemcpy(Y, X, N, type));  // NOLINT(caffe/alt_fn)
    }
  }

  void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {

    if (X != Y) {
      BOOL_CHECK(caffe::OpenCL::clMemcpy(Y, X, N, caffe::OpenCL::COPY_DEFAULT)); // NOLINT(caffe/alt_fn)
    }
  }

  template<typename T>
  void caffe_gpu_memset(const size_t Bytes, const T alpha, void* X) {
    try {
      caffe::OpenCL::clMemset(X, alpha, Bytes);
    } catch (std::exception &e) {
      LOG(ERROR)<<e.what();
    }
  }
  template void caffe_gpu_memset<char>(const size_t Bytes, const char alpha, void* X);
  template void caffe_gpu_memset<int>(const size_t Bytes, const int alpha, void* X);
  template void caffe_gpu_memset<float>(const size_t Bytes, const float alpha, void* X);
  template void caffe_gpu_memset<double>(const size_t Bytes, const double alpha, void* X);

  template<typename T>
  void caffe_gpu_asum(const int n, const T* x, T* y) {
    BOOL_CHECK(caffe::OpenCL::clBLASasum<T>(n, x, y));
  }
  template void caffe_gpu_asum<float>(const int n, const float* x, float* y);
  template void caffe_gpu_asum<double>(const int n, const double* x, double* y);

  template<typename T>
  void caffe_gpu_sign(const int n, const T* x, T* y) {
    BOOL_CHECK(caffe::OpenCL::clsign<T>(n, x, y));
  }
  template void caffe_gpu_sign<float>(const int n, const float* x, float* y);
  template void caffe_gpu_sign<double>(const int n, const double* x, double* y);

  template<typename T>
  void caffe_gpu_sgnbit(const int n, const T* x, T* y) {
    BOOL_CHECK(caffe::OpenCL::clsgnbit<T>(n, x, y));
  }
  template void caffe_gpu_sgnbit<float>(const int n, const float* x, float* y);
  template void caffe_gpu_sgnbit<double>(const int n, const double* x, double* y);

  template<typename T>
  void caffe_gpu_abs(const int n, const T* x, T* y) {
    BOOL_CHECK(caffe::OpenCL::clabs<T>(n, x, y));
  }
  template void caffe_gpu_abs<float>(const int n, const float* x, float* y);
  template void caffe_gpu_abs<double>(const int n, const double* x, double* y);

  template<typename T>
  void caffe_gpu_div(const int n, const T* x, const T* y, T* z) {
    BOOL_CHECK(caffe::OpenCL::cldiv<T>(n, x, y, z));
  }
  template void caffe_gpu_div<float>(const int n, const float* x, const float* y, float* z);
  template void caffe_gpu_div<double>(const int n, const double* x, const double* y, double* z);

  template<typename T>
  void caffe_gpu_mul(const int n, const T* x, const T* y, T* z) {
    BOOL_CHECK(caffe::OpenCL::clmul<T>(n, x, y, z));
  }
  template void caffe_gpu_mul<float>(const int n, const float* x, const float* y, float* z);
  template void caffe_gpu_mul<double>(const int n, const double* x, const double* y, double* z);

  template<typename T>
  void caffe_gpu_scale(const int n, const float alpha, const T* x, T* y) {
    BOOL_CHECK(caffe::OpenCL::clBLASscal<T>(n, alpha, x, y));
  }
  template void caffe_gpu_scale<float>(const int n, const float alpha, const float* x, float* y);
  template void caffe_gpu_scale<double>(const int n, const float alpha, const double* x, double* y);

  template<typename T>
  void caffe_gpu_set(const int N, const T alpha, T* Y) {

    try {
      caffe::OpenCL::clMemset(Y, alpha, sizeof(T) * N);
    } catch (std::exception &e) {
      LOG(ERROR)<<e.what();
    }
  }
  template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
  template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
  template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

  template<typename T>
  void caffe_gpu_dot(const int n, const T* x, const T* y, T* out) {
    BOOL_CHECK(caffe::OpenCL::clBLASdot<T>(n, x, 1, y, 1, out));
  }
  template void caffe_gpu_dot<float>(const int n, const float* x, const float* y, float* out);
  template void caffe_gpu_dot<double>(const int n, const double* x, const double* y, double* out);

  template<typename T>
  void caffe_gpu_sub(const int n, const T* a, const T* b, T* y) {
    BOOL_CHECK(caffe::OpenCL::clsub<T>(n, a, b, y));
  }
  template void caffe_gpu_sub<float>(const int n, const float* a, const float* b, float* y);
  template void caffe_gpu_sub<double>(const int n, const double* a, const double* b, double* y);

  template<typename T>
  void caffe_gpu_add(const int n, const T* a, const T* b, T* y) {
    BOOL_CHECK(caffe::OpenCL::cladd<T>(n, a, b, y));
  }
  template void caffe_gpu_add<float>(const int n, const float* a, const float* b, float* y);
  template void caffe_gpu_add<double>(const int n, const double* a, const double* b, double* y);

  template<typename T>
  void caffe_gpu_add_scalar(const int N, const T alpha, T* Y) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    //add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, Y);
    BOOL_CHECK(caffe::OpenCL::cladd_scalar(N, alpha, Y));
  }
  template void caffe_gpu_add_scalar<float>(const int N, const float alpha, float* Y);
  template void caffe_gpu_add_scalar<double>(const int N, const double alpha, double* Y);

  template<typename T>
  void caffe_gpu_powx(const int n, const T* a, const T alpha, T* y) {
    BOOL_CHECK(caffe::OpenCL::clpowx<T>(n, a, alpha, y));
  }
  template void caffe_gpu_powx<float>(const int n, const float* a, const float alpha, float* y);
  template void caffe_gpu_powx<double>(const int n, const double* a, const double alpha, double* y);

  template<typename T>
  void caffe_gpu_exp(const int n, const T* a, T* y) {
    BOOL_CHECK(caffe::OpenCL::clexp<T>(n, a, y));
  }
  template void caffe_gpu_exp<float>(const int n, const float* a, float* y);
  template void caffe_gpu_exp<double>(const int n, const double* a, double* y);

  template<typename T>
  void caffe_gpu_gemv(
      const CBLAS_TRANSPOSE TransA,
      const int m,
      const int n,
      const T alpha,
      const T* A,
      const T* x,
      const T beta,
      T* y) {

    clblasTranspose clTransA;
    switch (TransA) {
      case CblasNoTrans:
        clTransA = clblasNoTrans;
        break;
      case CblasTrans:
        clTransA = clblasTrans;
        break;
      case CblasConjTrans:
        clTransA = clblasConjTrans;
        break;
      default:
        LOG(ERROR)<< "unknown transpose mode.";
        return;
      }
#ifdef USE_CLGEMM
    BOOL_CHECK(caffe::OpenCL::clgemv<T>(clTransA, m, n, alpha, A, x, beta, y));
#else
    BOOL_CHECK(caffe::OpenCL::clBLASgemv<T>(clTransA, m, n, alpha, A, x, beta, y));
#endif
  }
  template void caffe_gpu_gemv<float>(
      const CBLAS_TRANSPOSE TransA,
      const int m,
      const int n,
      const float alpha,
      const float* A,
      const float* x,
      const float beta,
      float* y);
  template void caffe_gpu_gemv<double>(
      const CBLAS_TRANSPOSE TransA,
      const int m,
      const int n,
      const double alpha,
      const double* A,
      const double* x,
      const double beta,
      double* y);

  template<typename T>
  void caffe_gpu_gemv(
      const CBLAS_TRANSPOSE TransA,
      const int m,
      const int n,
      const T alpha,
      const T* A,
      const int step_A,
      const T* x,
      const int step_x,
      const T beta,
      T* y,
      const int step_y) {

    clblasTranspose clTransA;
    switch (TransA) {
      case CblasNoTrans:
        clTransA = clblasNoTrans;
        break;
      case CblasTrans:
        clTransA = clblasTrans;
        break;
      case CblasConjTrans:
        clTransA = clblasConjTrans;
        break;
      default:
        LOG(ERROR)<< "unknown transpose mode.";
        return;
      }

#ifdef USE_CLGEMM
    BOOL_CHECK(caffe::OpenCL::clgemv<T>(clTransA, m, n, alpha, A, step_A, x, step_x, beta, y, step_y));
#else
    BOOL_CHECK(caffe::OpenCL::clBLASgemv<T>(clTransA, m, n, alpha, A, step_A, x, step_x, beta, y, step_y));
#endif
  }

  template void caffe_gpu_gemv<float>(
      const CBLAS_TRANSPOSE TransA,
      const int m,
      const int n,
      const float alpha,
      const float* A,
      const int step_A,
      const float* x,
      const int step_x,
      const float beta,
      float* y,
      const int step_y);
  template void caffe_gpu_gemv<double>(
      const CBLAS_TRANSPOSE TransA,
      const int m,
      const int n,
      const double alpha,
      const double* A,
      const int step_A,
      const double* x,
      const int step_x,
      const double beta,
      double* y,
      const int step_y);

  template<typename T>
  void caffe_gpu_gemm(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const T alpha,
      const T* A,
      const T* B,
      const T beta,
      T* C,
      cl_event* event) {

    clblasTranspose clTransA;
    switch (TransA) {
      case CblasNoTrans:
        clTransA = clblasNoTrans;
        break;
      case CblasTrans:
        clTransA = clblasTrans;
        break;
      case CblasConjTrans:
        clTransA = clblasConjTrans;
        break;
      default:
        LOG(ERROR)<< "unknown transpose mode.";
        return;
      }

    clblasTranspose clTransB;
    switch (TransB) {
      case CblasNoTrans:
        clTransB = clblasNoTrans;
        break;
      case CblasTrans:
        clTransB = clblasTrans;
        break;
      case CblasConjTrans:
        clTransB = clblasConjTrans;
        break;
      default:
        LOG(ERROR)<< "unknown transpose mode.";
        return;
      }

#ifdef USE_CLGEMM
    BOOL_CHECK(caffe::OpenCL::clgemm<T>(clTransA, clTransB, M, N, K, alpha, A, B, beta, C, event));
#else
    BOOL_CHECK(caffe::OpenCL::clBLASgemm<T>(clTransA, clTransB, M, N, K, alpha, A, B, beta, C));
#endif
 }
  template void caffe_gpu_gemm<float>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const float alpha,
      const float* A,
      const float* B,
      const float beta,
      float* C,
      cl_event* event);
  template void caffe_gpu_gemm<double>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const double alpha,
      const double* A,
      const double* B,
      const double beta,
      double* C,
      cl_event* event);

  template<typename T>
  void caffe_gpu_gemm(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const T alpha,
      const T* A,
      const T* B,
      const T beta,
      T* C
      ) {

    return caffe_gpu_gemm<T>(TransA, TransB, M, N, K, alpha, A, B, beta, C, NULL);
  }
  template void caffe_gpu_gemm<float>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const float alpha,
      const float* A,
      const float* B,
      const float beta,
      float* C);
  template void caffe_gpu_gemm<double>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const double alpha,
      const double* A,
      const double* B,
      const double beta,
      double* C);

  template<typename T>
  void caffe_gpu_gemm(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const T alpha,
      const T* A,
      const size_t idx_offset_A,
      const T* B,
      const size_t idx_offset_B,
      const T beta,
      T* C,
      const size_t idx_offset_C,
      cl_event* event) {

    clblasTranspose clTransA;
    switch (TransA) {
      case CblasNoTrans:
        clTransA = clblasNoTrans;
        break;
      case CblasTrans:
        clTransA = clblasTrans;
        break;
      case CblasConjTrans:
        clTransA = clblasConjTrans;
        break;
      default:
        LOG(ERROR)<< "unknown transpose mode.";
        return;
      }

    clblasTranspose clTransB;
    switch (TransB) {
      case CblasNoTrans:
        clTransB = clblasNoTrans;
        break;
      case CblasTrans:
        clTransB = clblasTrans;
        break;
      case CblasConjTrans:
        clTransB = clblasConjTrans;
        break;
      default:
        LOG(ERROR)<< "unknown transpose mode.";
        return;
      }

#ifdef USE_CLGEMM
    BOOL_CHECK(caffe::OpenCL::clgemm<T>(clTransA, clTransB, M, N, K, alpha, A, idx_offset_A, B, idx_offset_B, beta, C, idx_offset_C, event));
#else
    BOOL_CHECK(caffe::OpenCL::clBLASgemm<T>(clTransA, clTransB, M, N, K, alpha, A, idx_offset_A, B, idx_offset_B, beta, C, idx_offset_C, event));
#endif
  }
  template void caffe_gpu_gemm<float>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const float alpha,
      const float* A,
      const size_t idx_offset_A,
      const float* B,
      const size_t idx_offset_B,
      const float beta,
      float* C,
      const size_t idx_offset_C,
      cl_event* event);
  template void caffe_gpu_gemm<double>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const double alpha,
      const double* A,
      const size_t idx_offset_A,
      const double* B,
      const size_t idx_offset_B,
      const double beta,
      double* C,
      const size_t idx_offset_C,
      cl_event* event);

  template<typename T>
  void caffe_gpu_gemm(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const T alpha,
      const T* A,
      const size_t idx_offset_A,
      const size_t lda,
      const T* B,
      const size_t idx_offset_B,
      const size_t ldb,
      const T beta,
      T* C,
      const size_t idx_offset_C,
      const size_t ldc,
      cl_event* event) {

    clblasTranspose clTransA;
    switch (TransA) {
      case CblasNoTrans:
        clTransA = clblasNoTrans;
        break;
      case CblasTrans:
        clTransA = clblasTrans;
        break;
      case CblasConjTrans:
        clTransA = clblasConjTrans;
        break;
      default:
        LOG(ERROR)<< "unknown transpose mode.";
        return;
      }

    clblasTranspose clTransB;
    switch (TransB) {
      case CblasNoTrans:
        clTransB = clblasNoTrans;
        break;
      case CblasTrans:
        clTransB = clblasTrans;
        break;
      case CblasConjTrans:
        clTransB = clblasConjTrans;
        break;
      default:
        LOG(ERROR)<< "unknown transpose mode.";
        return;
      }

#ifdef USE_CLGEMM
    BOOL_CHECK(caffe::OpenCL::clgemm<T>(clTransA, clTransB, M, N, K, alpha, A, idx_offset_A, B, idx_offset_B, beta, C, idx_offset_C, event));
#else
    BOOL_CHECK(caffe::OpenCL::clBLASgemm<T>(clTransA, clTransB, M, N, K, alpha, A, idx_offset_A, lda, B, idx_offset_B, ldb, beta, C, idx_offset_C, ldc, event));
#endif
  }
  template void caffe_gpu_gemm<float>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const float alpha,
      const float* A,
      const size_t idx_offset_A,
      const size_t lda,
      const float* B,
      const size_t idx_offset_B,
      const size_t ldb,
      const float beta,
      float* C,
      const size_t idx_offset_C,
      const size_t ldc,
      cl_event* event);
  template void caffe_gpu_gemm<double>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const double alpha,
      const double* A,
      const size_t idx_offset_A,
      const size_t lda,
      const double* B,
      const size_t idx_offset_B,
      const size_t ldb,
      const double beta,
      double* C,
      const size_t idx_offset_C,
      const size_t ldc,
      cl_event* event);


  template<typename T>
  void caffe_gpu_gemm(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const T alpha,
      const T* A,
      const size_t idx_offset_A,
      const T* B,
      const size_t idx_offset_B,
      const T beta,
      T* C,
      const size_t idx_offset_C
      ) {

      return caffe_gpu_gemm<T>(TransA, TransB, M, N, K, alpha, A, idx_offset_A, B, idx_offset_B, beta, C, idx_offset_C, NULL);
  }
  template void caffe_gpu_gemm<float>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const float alpha,
      const float* A,
      const size_t idx_offset_A,
      const float* B,
      const size_t idx_offset_B,
      const float beta,
      float* C,
      const size_t idx_offset_C);
  template void caffe_gpu_gemm<double>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const double alpha,
      const double* A,
      const size_t idx_offset_A,
      const double* B,
      const size_t idx_offset_B,
      const double beta,
      double* C,
      const size_t idx_offset_C);

  template<typename T>
  void caffe_gpu_gemm(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const T alpha,
      const T* A,
      const size_t idx_offset_A,
      const size_t lda,
      const T* B,
      const size_t idx_offset_B,
      const size_t ldb,
      const T beta,
      T* C,
      const size_t idx_offset_C,
      const size_t ldc
      ) {

      return caffe_gpu_gemm<T>(TransA, TransB, M, N, K, alpha, A, idx_offset_A, lda, B, idx_offset_B, ldb, beta, C, idx_offset_C, ldc, NULL);
  }
  template void caffe_gpu_gemm<float>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const float alpha,
      const float* A,
      const size_t idx_offset_A,
      const size_t lda,
      const float* B,
      const size_t idx_offset_B,
      const size_t ldb,
      const float beta,
      float* C,
      const size_t idx_offset_C,
      const size_t ldc);
  template void caffe_gpu_gemm<double>(
      const CBLAS_TRANSPOSE TransA,
      const CBLAS_TRANSPOSE TransB,
      const int M,
      const int N,
      const int K,
      const double alpha,
      const double* A,
      const size_t idx_offset_A,
      const size_t lda,
      const double* B,
      const size_t idx_offset_B,
      const size_t ldb,
      const double beta,
      double* C,
      const size_t idx_offset_C,
      const size_t ldc);

  template<typename T>
    void caffe_gpu_group_gemm(
        const CBLAS_TRANSPOSE TransA,
        const CBLAS_TRANSPOSE TransB,
        const int M, const int N, const int K,
        const int GM, const int GN, const int GK,
        const T alpha,
        const T* A,
        const T* B,
        const T beta,
        T* C) {

      clblasTranspose clTransA;
      switch (TransA) {
        case CblasNoTrans:
          clTransA = clblasNoTrans;
          break;
        case CblasTrans:
          clTransA = clblasTrans;
          break;
        case CblasConjTrans:
          clTransA = clblasConjTrans;
          break;
        default:
          LOG(ERROR)<< "unknown transpose mode.";
          return;
        }

      clblasTranspose clTransB;
      switch (TransB) {
        case CblasNoTrans:
          clTransB = clblasNoTrans;
          break;
        case CblasTrans:
          clTransB = clblasTrans;
          break;
        case CblasConjTrans:
          clTransB = clblasConjTrans;
          break;
        default:
          LOG(ERROR)<< "unknown transpose mode.";
          return;
        }

      BOOL_CHECK(caffe::OpenCL::cl_group_gemm<T>(clTransA, clTransB, M, N, K, GM, GN, GK, alpha, A, 0, B, 0, beta, C, 0, NULL));
    }
    template void caffe_gpu_group_gemm<float>(
        const CBLAS_TRANSPOSE TransA,
        const CBLAS_TRANSPOSE TransB,
        const int M, const int N, const int K,
        const int GM, const int GN, const int GK,
       const float alpha,
        const float* A,
        const float* B,
        const float beta,
        float* C);
    template void caffe_gpu_group_gemm<double>(
        const CBLAS_TRANSPOSE TransA,
        const CBLAS_TRANSPOSE TransB,
        const int M, const int N, const int K,
        const int GM, const int GN, const int GK,
        const double alpha,
        const double* A,
        const double* B,
        const double beta,
        double* C);

  template<typename T>
  void caffe_gpu_axpy(const int N, const T alpha, const T* X, T* Y) {

    TIME("clBLASaxpy()", {
        BOOL_CHECK(caffe::OpenCL::clBLASaxpy<T>(N, alpha, X, 1, Y, 1));
    });
  }
  template void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X, float* Y);
  template void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X, double* Y);

  template<typename T>
  void caffe_gpu_axpby(const int N, const T alpha, const T* X, const T beta, T* Y) {

    TIME("caffe_gpu_scal()", {
        caffe_gpu_scal<T>(N, beta, Y);
    });
    TIME("caffe_gpu_axpy()", {
        caffe_gpu_axpy<T>(N, alpha, X, Y);
    });
  }
  template void caffe_gpu_axpby<float>(
      const int N,
      const float alpha,
      const float* X,
      const float beta,
      float* Y);
  template void caffe_gpu_axpby<double>(
      const int N,
      const double alpha,
      const double* X,
      const double beta,
      double* Y);

  template<typename T>
  void caffe_gpu_scal(const int N, const T alpha, T *X) {

    //CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
    BOOL_CHECK(caffe::OpenCL::clBLASscal<T>(N, alpha, X, X));
  }
  template void caffe_gpu_scal<float>(const int N, const float alpha, float *X);
  template void caffe_gpu_scal<double>(const int N, const double alpha, double *X);

  void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
    BOOL_CHECK(caffe::OpenCL::cl_caffe_gpu_rng_uniform(n, r));
  }

  template<typename T>
  void caffe_gpu_rng_uniform(const int n, const T a, const T b, T* r) {
    BOOL_CHECK(caffe::OpenCL::cl_caffe_gpu_rng_uniform(n, a, b, r));
  }
  template void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b, float* r);
  template void caffe_gpu_rng_uniform<double>(
      const int n,
      const double a,
      const double b,
      double* r);

  template<typename T>
  void caffe_gpu_rng_gaussian(const int n, const T mu, const T sigma, T* r) {
    BOOL_CHECK(caffe::OpenCL::cl_caffe_gpu_rng_gaussian(n, mu, sigma, r));
  }
  template void caffe_gpu_rng_gaussian<float>(
      const int n,
      const float mu,
      const float sigma,
      float* r);
  template void caffe_gpu_rng_gaussian<double>(
      const int n,
      const double mu,
      const double sigma,
      double* r);

  template<typename T1, typename T2>
  void caffe_gpu_rng_bernoulli(const int n, const T1 p, T2* r) {
    BOOL_CHECK(caffe::OpenCL::cl_caffe_gpu_rng_bernoulli(n, p, r));
  }
  template void caffe_gpu_rng_bernoulli<float, int>(const int n, const float p, int* r);
  template void caffe_gpu_rng_bernoulli<double, int>(const int n, const double p, int* r);
  template void caffe_gpu_rng_bernoulli<float, unsigned int>(
      const int n,
      const float p,
      unsigned int* r);
  template void caffe_gpu_rng_bernoulli<double, unsigned int>(
      const int n,
      const double p,
      unsigned int* r);

}  // namespace caffe

#endif // USE_OPENCL
