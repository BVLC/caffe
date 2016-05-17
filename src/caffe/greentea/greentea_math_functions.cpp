/*
 * greentea_math_functions.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "caffe/common.hpp"
#include "caffe/device.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"

#include "caffe/util/math_functions.hpp"

#if defined(USE_CLBLAS)
  #include <clBLAS.h>       // NOLINT
#elif defined(USE_CLBLAST)
  #include <clblast.h>      // NOLINT
#else
  #include "viennacl/linalg/inner_prod.hpp"
  #include "viennacl/linalg/norm_1.hpp"
  #include "viennacl/linalg/norm_2.hpp"
  #include "viennacl/linalg/norm_inf.hpp"
  #include "viennacl/linalg/prod.hpp"
  #include "viennacl/matrix.hpp"
  #include "viennacl/scalar.hpp"
  #include "viennacl/vector.hpp"
#endif

// ViennaCL 1.5.1 compability fix
#ifndef VIENNACL_MINOR_VERSION
#define VIENNACL_MINOR_VERSION 5
#endif

#if VIENNACL_MINOR_VERSION > 5
#define VCL_ROW_MAJOR , true
#define VCL_COL_MAJOR , false
#else
#define VCL_ROW_MAJOR
#define VCL_COL_MAJOR
#endif

namespace caffe {

void greentea_memset(const int_tp ctx_id, const uint_tp N, const int_tp alpha,
                     cl_mem X, const int_tp offX) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  // OpenCL Version >= 1.2 approach
  // clEnqueueFillBuffer(ctx.get_queue().handle().get(),
  //  X, &alpha, sizeof(int_tp),
  //                     offX, N, 0, NULL, NULL);
  // OpenCL Version < 1.2 fallback
  typedef float Dtype;
  viennacl::ocl::kernel &oclk_fill = program.get_kernel(
      CL_KERNEL_SELECT("fillbuffer"));
  viennacl::ocl::enqueue(
      oclk_fill(static_cast<int_tp>(N), static_cast<unsigned char>(alpha),
                WrapHandle(X, &ctx), offX),
      ctx.get_queue());
}

// Copy from OpenCL buffer to main memory
void greentea_gpu_memcpy(const uint_tp N, const cl_mem X, const int_tp offX,
                         void *Y, viennacl::ocl::context *ctx) {
  if (Y != NULL) {
    clEnqueueReadBuffer(ctx->get_queue().handle().get(), X, CL_TRUE, offX, N, Y,
                        0,
                        NULL,
                        NULL);
  }
}

// Copy from main memory to OpenCL buffer
void greentea_gpu_memcpy(const uint_tp N, const void* X, cl_mem Y,
                         const int_tp offY, viennacl::ocl::context *ctx) {
  if (X != NULL) {
    clEnqueueWriteBuffer(ctx->get_queue().handle().get(), Y,
    CL_TRUE,
                         offY, N, X, 0, NULL, NULL);
  }
}

// Copy from OpenCL to OpenCL buffer
void greentea_gpu_memcpy(const uint_tp N, const cl_mem X, const int_tp offX,
                         cl_mem Y, const int_tp offY,
                         viennacl::ocl::context *ctx) {
  clEnqueueCopyBuffer(ctx->get_queue().handle().get(), X, Y, offX, offY, N, 0,
  NULL,
                      NULL);
}

template<typename Dtype>
void greentea_copy(const int_tp N, const cl_mem X, const int_tp offX, Dtype* Y,
                   viennacl::ocl::context *ctx) {
  greentea_gpu_memcpy(sizeof(Dtype) * N, X, offX * sizeof(Dtype), Y, ctx);
}

template<typename Dtype>
void greentea_copy(const int_tp N, const Dtype* X, cl_mem Y, const int_tp offY,
                   viennacl::ocl::context *ctx) {
  greentea_gpu_memcpy(sizeof(Dtype) * N, X, Y, offY * sizeof(Dtype), ctx);
}

// Copy from OpenCL buffer to OpenCL buffer
template<typename Dtype>
void greentea_copy(const int_tp N, const cl_mem X, const int_tp offX, cl_mem Y,
                   const int_tp offY, viennacl::ocl::context *ctx) {
  greentea_gpu_memcpy(sizeof(Dtype) * N, X, offX * sizeof(Dtype), Y,
                      offY * sizeof(Dtype), ctx);
}

// Explicit instantiations
template void greentea_copy<int_tp>(const int_tp N, const cl_mem X,
                                    const int_tp offX,
                                    int_tp* Y,
                                    viennacl::ocl::context *ctx);
template void greentea_copy<uint_tp>(const int_tp N, const cl_mem X,
                                     const int_tp offX, uint_tp* Y,
                                     viennacl::ocl::context *ctx);
template void greentea_copy<float>(const int_tp N, const cl_mem X,
                                   const int_tp offX, float* Y,
                                   viennacl::ocl::context *ctx);
template void greentea_copy<double>(const int_tp N, const cl_mem X,
                                    const int_tp offX, double* Y,
                                    viennacl::ocl::context *ctx);
template void greentea_copy<int_tp>(const int_tp N, const int_tp* X, cl_mem Y,
                                    const int_tp offY,
                                    viennacl::ocl::context *ctx);
template void greentea_copy<uint_tp>(const int_tp N, const uint_tp* X, cl_mem Y,
                                     const int_tp offY,
                                     viennacl::ocl::context *ctx);
template void greentea_copy<float>(const int_tp N, const float* X, cl_mem Y,
                                   const int_tp offY,
                                   viennacl::ocl::context *ctx);
template void greentea_copy<double>(const int_tp N, const double* X, cl_mem Y,
                                    const int_tp offY,
                                    viennacl::ocl::context *ctx);
template void greentea_copy<int_tp>(const int_tp N, const cl_mem X,
                                    const int_tp offX, cl_mem Y,
                                    const int_tp offY,
                                    viennacl::ocl::context *ctx);
template void greentea_copy<uint_tp>(const int_tp N, const cl_mem X,
                                     const int_tp offX, cl_mem Y,
                                     const int_tp offY,
                                     viennacl::ocl::context *ctx);
template void greentea_copy<float>(const int_tp N, const cl_mem X,
                                   const int_tp offX, cl_mem Y,
                                   const int_tp offY,
                                   viennacl::ocl::context *ctx);
template void greentea_copy<double>(const int_tp N, const cl_mem X,
                                    const int_tp offX, cl_mem Y,
                                    const int_tp offY,
                                    viennacl::ocl::context *ctx);

template<typename Dtype>
void greentea_gpu_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const Dtype beta, cl_mem C,
                       const int_tp offC) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Aptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), A, true, CL_MAP_READ,
        sizeof(Dtype) * offA, sizeof(Dtype) * M * K, 0, NULL, NULL, NULL));
    Dtype* Bptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), B, true, CL_MAP_READ,
        sizeof(Dtype) * offB, sizeof(Dtype) * N * K, 0, NULL, NULL, NULL));
    Dtype* Cptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), C, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(Dtype) * offC, sizeof(Dtype) * M * N, 0, NULL, NULL, NULL));

    caffe_cpu_gemm<Dtype>(TransA, TransB, M, N, K, alpha, Aptr, Bptr, beta,
                          Cptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A, Aptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), B, Bptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), C, Cptr, 0, NULL,
    NULL);
  } else {
    int_tp lda = (TransA == CblasNoTrans) ? K : M;
    int_tp ldb = (TransB == CblasNoTrans) ? N : K;
    int_tp ldc = N;

#if defined(USE_CLBLAS)

    clblasOrder clOrder = clblasRowMajor;
    clblasTranspose clTransA =
    (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
    clblasTranspose clTransB =
    (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSgemm(clOrder, clTransA, clTransB,
              M, N, K, alpha, A, offA, lda, B, offB, ldb, beta,
              C, offC, ldc, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDgemm(clOrder, clTransA, clTransB,
              M, N, K, alpha, A, offA, lda, B, offB, ldb, beta,
              C, offC, ldc, 1, &queue, 0, NULL, NULL));
    }

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;
    clblast::Transpose b_transpose = (TransB == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Gemm<float>(
          layout, a_transpose, b_transpose,
          M, N, K,
          alpha,
          A, offA, lda,
          B, offB, ldb,
          beta,
          C, offC, ldc,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Gemm<double>(
          layout, a_transpose, b_transpose,
          M, N, K,
          alpha,
          A, offA, lda,
          B, offB, ldb,
          beta,
          C, offC, ldc,
          &queue));
    }

#else  // default (ViennaCL)

    typedef typename viennacl::matrix_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::matrix_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    size_type A_size1 = static_cast<size_type>((TransA == CblasTrans) ? K : M);
    size_type A_size2 = static_cast<size_type>((TransA == CblasTrans) ? M : K);

    size_type B_size1 = static_cast<size_type>((TransB == CblasTrans) ? N : K);
    size_type B_size2 = static_cast<size_type>((TransB == CblasTrans) ? K : N);

    viennacl::matrix_base<Dtype, size_t, ptrdiff_t> matA(A, ctx, A_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(M), A_size2,
                                                       size_type(offA),
                                                       difference_type(1),
                                                       size_type(lda)
                                                       VCL_ROW_MAJOR);

    viennacl::matrix_base<Dtype, size_t, ptrdiff_t> matB(B, ctx, B_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(K), B_size2,
                                                       size_type(offB),
                                                       difference_type(1),
                                                       size_type(ldb)
                                                       VCL_ROW_MAJOR);

    viennacl::matrix_base<Dtype, size_t, ptrdiff_t> matC(C, ctx, size_type(M),
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(M),
                                                       size_type(N),
                                                       size_type(offC),
                                                       difference_type(1),
                                                       size_type(ldc)
                                                       VCL_ROW_MAJOR);

    if (TransA == CblasTrans && TransB == CblasTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), viennacl::trans(matB),
                                  matC, alpha, beta);
    else if (TransA == CblasTrans && TransB == CblasNoTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), matB, matC, alpha,
                                  beta);
    else if (TransA == CblasNoTrans && TransB == CblasTrans)
      viennacl::linalg::prod_impl(matA, viennacl::trans(matB), matC, alpha,
                                  beta);
    else if (TransA == CblasNoTrans && TransB == CblasNoTrans)
      viennacl::linalg::prod_impl(matA, matB, matC, alpha, beta);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_gemm<float>(const int_tp ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int_tp M, const int_tp N,
                                       const int_tp K, const float alpha,
                                       const cl_mem A, const int_tp offA,
                                       const cl_mem B, const int_tp offB,
                                       const float beta, cl_mem C,
                                       const int_tp offC);
template void greentea_gpu_gemm<double>(const int_tp ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const CBLAS_TRANSPOSE TransB,
                                        const int_tp M, const int_tp N,
                                        const int_tp K, const double alpha,
                                        const cl_mem A, const int_tp offA,
                                        const cl_mem B, const int_tp offB,
                                        const double beta, cl_mem C,
                                        const int_tp offC);

template<typename Dtype>
void greentea_gpu_gemv(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const int_tp M, const int_tp N, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem x,
                       const int_tp offx, const Dtype beta, cl_mem y,
                       const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Aptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), A, true, CL_MAP_READ,
        sizeof(Dtype) * offA, sizeof(Dtype) * M * N, 0, NULL, NULL, NULL));
    Dtype* xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x, true, CL_MAP_READ,
        sizeof(Dtype) * offx, sizeof(Dtype) * (TransA == CblasTrans) ? M : N, 0,
        NULL,
        NULL, NULL));
    Dtype* yptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), y, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(Dtype) * offy, sizeof(Dtype) * (TransA == CblasTrans) ? N : M, 0,
        NULL,
        NULL, NULL));

    caffe_cpu_gemv<Dtype>(TransA, M, N, alpha, Aptr, xptr, beta, yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A, Aptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x, xptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), y, yptr, 0, NULL,
    NULL);
  } else {
#if defined(USE_CLBLAS)

    clblasTranspose clTransA =
    (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSgemv(clblasRowMajor,
              clTransA, M, N, alpha, A, offA, N, x, offx, 1,
              beta, y, offy, 1, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDgemv(clblasRowMajor,
              clTransA, M, N, alpha, A, offA, N, x, offx, 1,
              beta, y, offy, 1, 1, &queue, 0, NULL, NULL));
    }

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    const size_t ldA = N;
    const size_t incx = 1;
    const size_t incy = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Gemv<float>(
          layout, a_transpose,
          M, N,
          alpha,
          A, offA, ldA,
          x, offx, incx,
          beta,
          y, offy, incy,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Gemv<double>(
          layout, a_transpose,
          M, N,
          alpha,
          A, offA, ldA,
          x, offx, incx,
          beta,
          y, offy, incy,
          &queue));
    }

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(
        x, size_type((TransA == CblasTrans) ? M : N), size_type(offx),
        difference_type(1), ctx);
    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v2(
        y, size_type((TransA == CblasTrans) ? N : M), size_type(offy),
        difference_type(1), ctx);
    viennacl::matrix_base<Dtype, size_t, ptrdiff_t> mat(A, ctx, size_type(M),
                                                      size_type(0),
                                                      difference_type(1),
                                                      size_type(M),
                                                      size_type(N),
                                                      size_type(offA),
                                                      difference_type(1),
                                                      size_type(N)
                                                      VCL_ROW_MAJOR);
    v2 *= beta;
    if (TransA == CblasTrans) {
      v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
    } else {
      v2 += alpha * viennacl::linalg::prod(mat, v1);
    }

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_gemv<float>(const int_tp ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const int_tp M, const int_tp N,
                                       const float alpha, const cl_mem A,
                                       const int_tp offA, const cl_mem x,
                                       const int_tp offx, const float beta,
                                       cl_mem y, const int_tp offy);
template void greentea_gpu_gemv<double>(const int_tp ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const int_tp M, const int_tp N,
                                        const double alpha, const cl_mem A,
                                        const int_tp offA, const cl_mem x,
                                        const int_tp offx, const double beta,
                                        cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_axpy(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                       const cl_mem X, const int_tp offX, cl_mem Y,
                       const int_tp offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X, true, CL_MAP_READ,
        sizeof(Dtype) * offX, sizeof(Dtype) * N, 0, NULL, NULL, NULL));
    Dtype* Yptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y, true, CL_MAP_WRITE,
        sizeof(Dtype) * offY, sizeof(Dtype) * N, 0, NULL, NULL, NULL));

    caffe_axpy<Dtype>(N, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X, Xptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y, Yptr, 0, NULL,
    NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSaxpy(N, alpha, X, offX,
              1, Y, offY, 1, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDaxpy(N, alpha, X, offX,
              1, Y, offY, 1, 1, &queue, 0, NULL, NULL));
    }

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Axpy<float>(
          N,
          alpha,
          X, offX, incX,
          Y, offY, incY,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Axpy<double>(
          N,
          alpha,
          X, offX, incX,
          Y, offY, incY,
          &queue));
    }

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(X, size_type(N),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v2(Y, size_type(N),
                                                     size_type(offY),
                                                     difference_type(1), ctx);
    v2 += alpha * v1;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_axpy<float>(const int_tp ctx_id, const int_tp N,
                                       const float alpha, const cl_mem X,
                                       const int_tp offX, cl_mem Y,
                                       const int_tp offY);
template void greentea_gpu_axpy<double>(const int_tp ctx_id, const int_tp N,
                                        const double alpha, const cl_mem X,
                                        const int_tp offX, cl_mem Y,
                                        const int_tp offY);

template<typename Dtype>
void greentea_gpu_mul(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_mul = program.get_kernel(CL_KERNEL_SELECT("mul"));
  viennacl::ocl::enqueue(
      oclk_mul(N, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_mul<float>(const int_tp ctx_id, const int_tp N,
                                      const cl_mem a, const int_tp offa,
                                      const cl_mem b, const int_tp offb,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_mul<double>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       const cl_mem b, const int_tp offb,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_div(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_div = program.get_kernel(CL_KERNEL_SELECT("div"));
  viennacl::ocl::enqueue(
      oclk_div(N, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_div<float>(const int_tp ctx_id, const int_tp N,
                                      const cl_mem a, const int_tp offa,
                                      const cl_mem b, const int_tp offb,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_div<double>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       const cl_mem b, const int_tp offb,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_scal(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                       cl_mem x, int_tp offx) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), x, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(Dtype) * offx, sizeof(Dtype) * N, 0, NULL, NULL, NULL));

    caffe_scal<Dtype>(N, alpha, xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), x, xptr, 0, NULL,
    NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(clblasSscal(N, alpha, x, offx,
              1, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(clblasDscal(N, alpha, x, offx,
              1, 1, &queue, 0, NULL, NULL));
    }

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incx = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Scal<float>(
          N,
          alpha,
          x, offx, incx,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Scal<double>(
          N,
          alpha,
          x, offx, incx,
          &queue));
    }

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(x, size_type(N),
                                                     size_type(offx),
                                                     difference_type(1), ctx);
    v1 *= alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_scal<float>(const int_tp ctx_id, const int_tp N,
                                       const float alpha, cl_mem x,
                                       const int_tp offx);
template void greentea_gpu_scal<double>(const int_tp ctx_id, const int_tp N,
                                        const double alpha, cl_mem x,
                                        const int_tp offx);

template<typename Dtype>
void greentea_gpu_axpby(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                        const cl_mem X, const int_tp offX, const Dtype beta,
                        cl_mem Y, const int_tp offY) {
  greentea_gpu_scal<Dtype>(ctx_id, N, beta, Y, offY);
  greentea_gpu_axpy<Dtype>(ctx_id, N, alpha, X, offX, Y, offY);
}

template void greentea_gpu_axpby<float>(const int_tp ctx_id, const int_tp N,
                                        const float alpha, const cl_mem X,
                                        const int_tp offX, const float beta,
                                        cl_mem Y, const int_tp offY);

template void greentea_gpu_axpby<double>(const int_tp ctx_id, const int_tp N,
                                         const double alpha, const cl_mem X,
                                         const int_tp offX, const double beta,
                                         cl_mem Y, const int_tp offY);

template<typename Dtype>
void greentea_gpu_dot(const int_tp ctx_id, const int_tp n, const cl_mem X,
                      const int_tp offX, const cl_mem Y, const int_tp offY,
                      Dtype* out) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X, true, CL_MAP_READ,
        sizeof(Dtype) * offX, sizeof(Dtype) * n, 0, NULL, NULL, NULL));
    Dtype* Yptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y, true, CL_MAP_READ,
        sizeof(Dtype) * offY, sizeof(Dtype) * n, 0, NULL, NULL, NULL));

    *out = caffe_cpu_dot<Dtype>(n, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X, Xptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y, Yptr, 0, NULL,
    NULL);

  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        sizeof(Dtype), NULL, &err);
    cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        n * sizeof(Dtype), NULL, &err);

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSdot(n, gpuout, 0, X, offX, 1, Y,
              offY, 1, scratch, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDdot(n, gpuout, 0, X, offX, 1, Y,
              offY, 1, scratch, 1, &queue, 0, NULL, NULL));
    }

    greentea_gpu_memcpy(sizeof(Dtype), gpuout, 0, out, &ctx);

    clReleaseMemObject(gpuout);
    clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;
    cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(Dtype), NULL, &err);
    // TODO: error handling.

    const size_t offZ = 0;
    const size_t incX = 1;
    const size_t incY = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Dot<float>(
          n,
          Z, offZ,
          X, offX, incX,
          Y, offY, incY,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Dot<double>(
          n,
          Z, offZ,
          X, offX, incX,
          Y, offY, incY,
          &queue));
    }

    greentea_gpu_memcpy(sizeof(Dtype), Z, offZ, out, &ctx);
    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(X, size_type(n),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v2(Y, size_type(n),
                                                     size_type(offY),
                                                     difference_type(1), ctx);

    *out = viennacl::linalg::inner_prod(v1, v2);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_dot<float>(const int_tp ctx_id, const int_tp n,
                                      const cl_mem X, const int_tp offX,
                                      const cl_mem Y, const int_tp offY,
                                      float* out);
template void greentea_gpu_dot<double>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem X, const int_tp offX,
                                       const cl_mem Y, const int_tp offY,
                                       double* out);

template<typename Dtype>
void greentea_gpu_asum(const int_tp ctx_id, const int_tp n, const cl_mem X,
                       const int_tp offX, Dtype* Y) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X, true, CL_MAP_READ,
        sizeof(Dtype) * offX, sizeof(Dtype) * n, 0, NULL, NULL, NULL));

    *Y = caffe_cpu_asum<Dtype>(n, Xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X, Xptr, 0, NULL,
    NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        sizeof(Dtype), NULL, &err);
    cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
        n * sizeof(Dtype), NULL, &err);

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSasum(n, gpuout, 0, X, offX, 1,
              scratch, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDasum(n, gpuout, 0, X, offX, 1,
              scratch, 1, &queue, 0, NULL, NULL));
    }

    greentea_gpu_memcpy(sizeof(Dtype), gpuout, 0, Y, &ctx);

    clReleaseMemObject(gpuout);
    clReleaseMemObject(scratch);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;
    cl_mem Z = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
      sizeof(Dtype), NULL, &err);
    // TODO: error handling.

    const size_t offZ = 0;
    const size_t incX = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Asum<float>(
          n,
          Z, offZ,
          X, offX, incX,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Asum<double>(
          n,
          Z, offZ,
          X, offX, incX,
          &queue));
    }

    greentea_gpu_memcpy(sizeof(Dtype), Z, offZ, Y, &ctx);

    clReleaseMemObject(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(X, size_type(n),
                                                     size_type(offX),
                                                     difference_type(1), ctx);

    *Y = viennacl::linalg::norm_1(v1);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_asum<float>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem X, const int_tp offX,
                                       float* Y);
template void greentea_gpu_asum<double>(const int_tp ctx_id, const int_tp n,
                                        const cl_mem X, const int_tp offX,
                                        double* Y);

template<typename Dtype>
void greentea_gpu_scale(const int_tp ctx_id, const int_tp n, const Dtype alpha,
                        const cl_mem X, const int_tp offX, cl_mem Y,
                        const int_tp offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Xptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), X, true, CL_MAP_READ,
        sizeof(Dtype) * offX, sizeof(Dtype) * n, 0, NULL, NULL, NULL));
    Dtype* Yptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Y, true, CL_MAP_WRITE,
        sizeof(Dtype) * offY, sizeof(Dtype) * n, 0, NULL, NULL, NULL));

    caffe_cpu_scale<Dtype>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), X, Xptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Y, Yptr, 0, NULL,
    NULL);
  } else {
#if defined(USE_CLBLAS)

    // FIXME: Remove, as can reuse ctx obtained above?
    viennacl::ocl::context ctx = viennacl::ocl::get_context(ctx_id);
    cl_command_queue queue = ctx.get_queue().handle().get();

    // FIXME: Use xAXPY with beta = 0?
    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasScopy(n, X, offX, 1, Y, offY, 1, 1, &queue, 0, NULL, NULL));
      GREENTEA_CL_BLAS_CHECK(
          clblasSscal(n, alpha, Y, offY, 1, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDcopy(n, X, offX, 1, Y, offY, 1, 1, &queue, 0, NULL, NULL));
      GREENTEA_CL_BLAS_CHECK(
          clblasDscal(n, alpha, Y, offY, 1, 1, &queue, 0, NULL, NULL));
    }

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CLBLAST_CHECK(
        clblast::Copy<float>(
          n,
          X, offX, incX,
          Y, offY, incY,
          &queue));
      GREENTEA_CLBLAST_CHECK(
        clblast::Scal<float>(
          n,
          alpha,
          Y, offY, incY,
          &queue));
    } else {
      GREENTEA_CLBLAST_CHECK(
        clblast::Copy<double>(
          n,
          X, offX, incX,
          Y, offY, incY,
          &queue));
      GREENTEA_CLBLAST_CHECK(
        clblast::Scal<double>(
          n,
          alpha,
          Y, offY, incY,
          &queue));
    }

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v1(X, size_type(n),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<Dtype, size_t, ptrdiff_t> v2(Y, size_type(n),
                                                     size_type(offY),
                                                     difference_type(1), ctx);

    v2 = v1 * alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template void greentea_gpu_scale<float>(const int_tp ctx_id, const int_tp n,
                                        const float alpha, const cl_mem X,
                                        const int_tp offX, cl_mem Y,
                                        const int_tp offY);

template void greentea_gpu_scale<double>(const int_tp ctx_id, const int_tp n,
                                         const double alpha, const cl_mem X,
                                         const int_tp offX, cl_mem Y,
                                         const int_tp offY);

template<typename Dtype>
void greentea_gpu_set(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                      cl_mem Y, const int_tp offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();
  // OpenCL Version >= 1.2 approach
  // clEnqueueFillBuffer(ctx.get_queue().handle().get(),
  //                  Y, &alpha, sizeof(Dtype),
  //                  offY, N, 0, NULL, NULL);

  // OpenCL Version < 1.2 fallback
  viennacl::ocl::kernel &oclk_fill = program.get_kernel(
      CL_KERNEL_SELECT("fill"));
  viennacl::ocl::enqueue(oclk_fill(N, alpha, WrapHandle(Y, &ctx), offY),
                         ctx.get_queue());
}

template void greentea_gpu_set<int_tp>(const int_tp ctx_id, const int_tp N,
                                       const int_tp alpha, cl_mem Y,
                                       const int_tp offY);
template void greentea_gpu_set<float>(const int_tp ctx_id, const int_tp N,
                                      const float alpha, cl_mem Y,
                                      const int_tp offY);
template void greentea_gpu_set<double>(const int_tp ctx_id, const int_tp N,
                                       const double alpha, cl_mem Y,
                                       const int_tp offY);

template<typename Dtype>
void greentea_gpu_add_scalar(const int_tp ctx_id, const int_tp N,
                             const Dtype alpha, cl_mem Y, const int_tp offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_add_scalar = program.get_kernel(
      CL_KERNEL_SELECT("add_scalar"));
  viennacl::ocl::enqueue(oclk_add_scalar(N, alpha, WrapHandle(Y, &ctx), offY),
                         ctx.get_queue());
}

template void greentea_gpu_add_scalar<float>(const int_tp ctx_id,
                                             const int_tp N, const float alpha,
                                             cl_mem Y, const int_tp offY);
template void greentea_gpu_add_scalar<double>(const int_tp ctx_id,
                                              const int_tp N,
                                              const double alpha, cl_mem Y,
                                              const int_tp offY);

template<typename Dtype>
void greentea_gpu_add(const int_tp ctx_id, const int_tp n, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_add = program.get_kernel(CL_KERNEL_SELECT("add"));
  viennacl::ocl::enqueue(
      oclk_add(n, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_add<float>(const int_tp ctx_id, const int_tp n,
                                      const cl_mem a, const int_tp offa,
                                      const cl_mem b, const int_tp offb,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_add<double>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem a, const int_tp offa,
                                       const cl_mem b, const int_tp offb,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_sub(const int_tp ctx_id, const int_tp n, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sub = program.get_kernel(CL_KERNEL_SELECT("sub"));
  viennacl::ocl::enqueue(
      oclk_sub(n, WrapHandle(a, &ctx), offa, WrapHandle(b, &ctx), offb,
               WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_sub<float>(const int_tp ctx_id, const int_tp n,
                                      const cl_mem a, const int_tp offa,
                                      const cl_mem b, const int_tp offb,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_sub<double>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem a, const int_tp offa,
                                       const cl_mem b, const int_tp offb,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_abs(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_abs = program.get_kernel(CL_KERNEL_SELECT("abs"));
  viennacl::ocl::enqueue(
      oclk_abs(N, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_abs<float>(const int_tp ctx_id, const int_tp N,
                                      const cl_mem a, const int_tp offa,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_abs<double>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_exp(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_exp = program.get_kernel(CL_KERNEL_SELECT("exp"));
  viennacl::ocl::enqueue(
      oclk_exp(N, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_exp<float>(const int_tp ctx_id, const int_tp N,
                                      const cl_mem a, const int_tp offa,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_exp<double>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_powx(const int_tp ctx_id, const int_tp N, const cl_mem a,
                       const int_tp offa, const Dtype alpha, cl_mem y,
                       const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_powx = program.get_kernel(
      CL_KERNEL_SELECT("powx"));
  viennacl::ocl::enqueue(
      oclk_powx(N, WrapHandle(a, &ctx), offa, alpha, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_powx<float>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       const float alpha, cl_mem y,
                                       const int_tp offy);
template void greentea_gpu_powx<double>(const int_tp ctx_id, const int_tp N,
                                        const cl_mem a, const int_tp offa,
                                        const double alpha, cl_mem y,
                                        const int_tp offy);

template<typename Dtype>
void greentea_gpu_log(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_log = program.get_kernel(CL_KERNEL_SELECT("log"));
  viennacl::ocl::enqueue(
      oclk_log(N, WrapHandle(a, &ctx), offa, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_log<float>(const int_tp ctx_id, const int_tp N,
                                      const cl_mem a, const int_tp offa,
                                      cl_mem y, const int_tp offy);
template void greentea_gpu_log<double>(const int_tp ctx_id, const int_tp N,
                                       const cl_mem a, const int_tp offa,
                                       cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_sign(const int_tp ctx_id, const int_tp n, const cl_mem x,
int_tp offx,
                       cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sign = program.get_kernel(
      CL_KERNEL_SELECT("sign"));
  viennacl::ocl::enqueue(
      oclk_sign(n, WrapHandle(x, &ctx), offx, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_sign<float>(const int_tp ctx_id, const int_tp n,
                                       const cl_mem x, int_tp offx, cl_mem y,
                                       const int_tp offy);
template void greentea_gpu_sign<double>(const int_tp ctx_id, const int_tp n,
                                        const cl_mem x, int_tp offx, cl_mem y,
                                        const int_tp offy);

template<typename Dtype>
void greentea_gpu_sgnbit(const int_tp ctx_id, const int_tp n, const cl_mem x,
int_tp offx,
                         cl_mem y, const int_tp offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(ctx_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sgnbit = program.get_kernel(
      CL_KERNEL_SELECT("sgnbit"));
  viennacl::ocl::enqueue(
      oclk_sgnbit(n, WrapHandle(x, &ctx), offx, WrapHandle(y, &ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_sgnbit<float>(const int_tp ctx_id, const int_tp n,
                                         const cl_mem x, int_tp offx, cl_mem y,
                                         const int_tp offy);
template void greentea_gpu_sgnbit<double>(const int_tp ctx_id, const int_tp n,
                                          const cl_mem x, int_tp offx, cl_mem y,
                                          const int_tp offy);

void greentea_gpu_rng_uniform(const int_tp ctx_id, const int_tp n, cl_mem r,
int_tp offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<uint_tp> random(n);  //NOLINT
  caffe_rng_uniform(n, &random[0]);
  greentea_gpu_memcpy(sizeof(uint_tp) * n, &random[0], r, offr, &ctx);
}

template<typename Dtype>
void greentea_gpu_rng_uniform(const int_tp ctx_id, const int_tp n,
                              const Dtype a, const Dtype b, cl_mem r,
                              const int_tp offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<Dtype> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  greentea_gpu_memcpy(sizeof(Dtype) * n, &random[0], r, offr, &ctx);
}

template void greentea_gpu_rng_uniform<float>(const int_tp ctx_id,
                                              const int_tp n, const float a,
                                              const float b, cl_mem r,
                                              const int_tp offr);
template void greentea_gpu_rng_uniform<double>(const int_tp ctx_id,
                                               const int_tp n, const double a,
                                               const double b, cl_mem r,
                                               const int_tp offr);

template<typename Dtype>
void greentea_gpu_rng_gaussian(const int_tp ctx_id, const int_tp n,
                               const Dtype mu, const Dtype sigma, cl_mem r,
                               const int_tp offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<Dtype> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  greentea_gpu_memcpy(sizeof(Dtype) * n, &random[0], r, offr, &ctx);
}

template void greentea_gpu_rng_gaussian<float>(const int_tp ctx_id,
                                               const int_tp n, const float mu,
                                               const float sigma, cl_mem r,
                                               const int_tp offr);

template void greentea_gpu_rng_gaussian<double>(const int_tp ctx_id,
                                                const int_tp n, const double mu,
                                                const double sigma, cl_mem r,
                                                const int_tp offr);

}  // namespace caffe
#endif
