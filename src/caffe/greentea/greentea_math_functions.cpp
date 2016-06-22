/*
 * greentea_math_functions.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "caffe/common.hpp"
#include "caffe/device.hpp"

#ifdef USE_GREENTEA
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

#include "caffe/greentea/greentea.hpp"
#include "caffe/util/math_functions.hpp"

#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"

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

void caffe_gpu_memset(const uint_tp N, const int_tp alpha, void* X) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<uint8_t> bufX = clState.get_buffer_mem(X);

  cl_mem Mem_X = bufX.memobj;

  int offX = bufX.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
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
                WrapHandle(Mem_X, &ctx), offX),
      ctx.get_queue());
}

template<>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, const int_tp M,
                           const int_tp N, const int_tp K, const float alpha,
                           const float* A, const float* B, const float beta,
                           float* C) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufA = clState.get_buffer_mem(A);
  ClMemOff<float> bufB = clState.get_buffer_mem(B);
  ClMemOff<float> bufC = clState.get_buffer_mem(C);

  cl_mem Mem_A = bufA.memobj;
  cl_mem Mem_B = bufB.memobj;
  cl_mem Mem_C = bufC.memobj;

  int offA = bufA.offset;
  int offB = bufB.offset;
  int offC = bufC.offset;

  int dev_id = clState.get_mem_dev(Mem_A);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Aptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_A, true, CL_MAP_READ,
        sizeof(float) * offA, sizeof(float) * M * K, 0, NULL, NULL, NULL));
    float* Bptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_B, true, CL_MAP_READ,
        sizeof(float) * offB, sizeof(float) * N * K, 0, NULL, NULL, NULL));
    float* Cptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_C, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(float) * offC, sizeof(float) * M * N, 0, NULL, NULL, NULL));

    caffe_cpu_gemm<float>(TransA, TransB, M, N, K, alpha, Aptr, Bptr, beta,
                          Cptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_A, Aptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_B, Bptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_C, Cptr,
    0, NULL, NULL);
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

    GREENTEA_CL_BLAS_CHECK(
        clblasSgemm(clOrder, clTransA, clTransB,
            M, N, K, alpha, Mem_A, offA, lda, Mem_B, offB, ldb, beta,
            Mem_C, offC, ldc, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;
    clblast::Transpose b_transpose = (TransB == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    GREENTEA_CLBLAST_CHECK(
      clblast::Gemm<float>(
        layout, a_transpose, b_transpose,
        M, N, K,
        alpha,
        Mem_A, offA, lda,
        Mem_B, offB, ldb,
        beta,
        Mem_C, offC, ldc,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::matrix_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::matrix_base<float,
        uint_tp, int_tp>::size_type difference_type;

    size_type A_size1 = static_cast<size_type>((TransA == CblasTrans) ? K : M);
    size_type A_size2 = static_cast<size_type>((TransA == CblasTrans) ? M : K);

    size_type B_size1 = static_cast<size_type>((TransB == CblasTrans) ? N : K);
    size_type B_size2 = static_cast<size_type>((TransB == CblasTrans) ? K : N);

    viennacl::matrix_base<float, size_t, ptrdiff_t> matA(Mem_A, ctx,
                                                         A_size1,
                                                         size_type(0),
                                                         difference_type(1),
                                                         size_type(M),
                                                         A_size2,
                                                         size_type(offA),
                                                         difference_type(1),
                                                         size_type(lda)
                                                         VCL_ROW_MAJOR);

    viennacl::matrix_base<float, size_t, ptrdiff_t> matB(Mem_B, ctx,
                                                         B_size1,
                                                         size_type(0),
                                                         difference_type(1),
                                                         size_type(K), B_size2,
                                                         size_type(offB),
                                                         difference_type(1),
                                                         size_type(ldb)
                                                         VCL_ROW_MAJOR);

    viennacl::matrix_base<float, size_t, ptrdiff_t> matC(Mem_C, ctx,
                                                         size_type(M),
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

template<>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB, const int_tp M,
                            const int_tp N, const int_tp K, const double alpha,
                            const double* A, const double* B, const double beta,
                            double* C) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufA = clState.get_buffer_mem(A);
  ClMemOff<double> bufB = clState.get_buffer_mem(B);
  ClMemOff<double> bufC = clState.get_buffer_mem(C);

  cl_mem Mem_A = bufA.memobj;
  cl_mem Mem_B = bufB.memobj;
  cl_mem Mem_C = bufC.memobj;

  int offA = bufA.offset;
  int offB = bufB.offset;
  int offC = bufC.offset;

  int dev_id = clState.get_mem_dev(Mem_A);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Aptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_A, true, CL_MAP_READ,
        sizeof(double) * offA, sizeof(double) * M * K, 0, NULL, NULL, NULL));
    double* Bptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_B, true, CL_MAP_READ,
        sizeof(double) * offB, sizeof(double) * N * K, 0, NULL, NULL, NULL));
    double* Cptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_C, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(double) * offC, sizeof(double) * M * N, 0, NULL, NULL, NULL));

    caffe_cpu_gemm<double>(TransA, TransB, M, N, K, alpha, Aptr, Bptr, beta,
                          Cptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_A, Aptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_B, Bptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_C, Cptr,
    0, NULL, NULL);
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

    GREENTEA_CL_BLAS_CHECK(
        clblasDgemm(clOrder, clTransA, clTransB,
            M, N, K, alpha, Mem_A, offA, lda, Mem_B, offB, ldb, beta,
            Mem_C, offC, ldc, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;
    clblast::Transpose b_transpose = (TransB == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    GREENTEA_CLBLAST_CHECK(
      clblast::Gemm<double>(
        layout, a_transpose, b_transpose,
        M, N, K,
        alpha,
        Mem_A, offA, lda,
        Mem_B, offB, ldb,
        beta,
        Mem_C, offC, ldc,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::matrix_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::matrix_base<double,
        uint_tp, int_tp>::size_type difference_type;

    size_type A_size1 = static_cast<size_type>((TransA == CblasTrans) ? K : M);
    size_type A_size2 = static_cast<size_type>((TransA == CblasTrans) ? M : K);

    size_type B_size1 = static_cast<size_type>((TransB == CblasTrans) ? N : K);
    size_type B_size2 = static_cast<size_type>((TransB == CblasTrans) ? K : N);

    viennacl::matrix_base<double, size_t, ptrdiff_t> matA(Mem_A, ctx,
                                                          A_size1,
                                                          size_type(0),
                                                          difference_type(1),
                                                          size_type(M),
                                                          A_size2,
                                                          size_type(offA),
                                                          difference_type(1),
                                                          size_type(lda)
                                                          VCL_ROW_MAJOR);

    viennacl::matrix_base<double, size_t, ptrdiff_t> matB(Mem_B, ctx,
                                                          B_size1,
                                                          size_type(0),
                                                          difference_type(1),
                                                          size_type(K),
                                                          B_size2,
                                                          size_type(offB),
                                                          difference_type(1),
                                                          size_type(ldb)
                                                          VCL_ROW_MAJOR);

    viennacl::matrix_base<double, size_t, ptrdiff_t> matC(Mem_C, ctx,
                                                          size_type(M),
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

template<>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int_tp M,
                           const int_tp N, const float alpha, const float* A,
                           const float* x, const float beta, float* y) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufA = clState.get_buffer_mem(A);
  ClMemOff<float> bufx = clState.get_buffer_mem(x);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_A = bufA.memobj;
  cl_mem Mem_x = bufx.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offA = bufA.offset;
  int offx = bufx.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_A);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Aptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_A, true, CL_MAP_READ,
        sizeof(float) * offA, sizeof(float) * M * N, 0, NULL, NULL, NULL));
    float* xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_x, true, CL_MAP_READ,
        sizeof(float) * offx, sizeof(float) * (TransA == CblasTrans) ? M : N, 0,
        NULL,
        NULL, NULL));
    float* yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_y, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(float) * offy, sizeof(float) * (TransA == CblasTrans) ? N : M, 0,
        NULL,
        NULL, NULL));

    caffe_cpu_gemv<float>(TransA, M, N, alpha, Aptr, xptr, beta, yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_A, Aptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_x, xptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_y, yptr,
    0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    clblasTranspose clTransA =
    (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    GREENTEA_CL_BLAS_CHECK(
      clblasSgemv(clblasRowMajor,
            clTransA, M, N, alpha, Mem_A, offA, N, Mem_x, offx, 1,
            beta, Mem_y, offy, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    const size_t ldA = N;
    const size_t incx = 1;
    const size_t incy = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Gemv<float>(
        layout, a_transpose,
        M, N,
        alpha,
        Mem_A, offA, ldA,
        Mem_x, offx, incx,
        beta,
        Mem_y, offy, incy,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(
        Mem_x, size_type((TransA == CblasTrans) ? M : N), size_type(offx),
        difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(
        Mem_y, size_type((TransA == CblasTrans) ? N : M), size_type(offy),
        difference_type(1), ctx);
    viennacl::matrix_base<float, size_t, ptrdiff_t> mat(Mem_A, ctx,
                                                        size_type(M),
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

template<>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int_tp M,
                            const int_tp N, const double alpha, const double* A,
                            const double* x, const double beta, double* y) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufA = clState.get_buffer_mem(A);
  ClMemOff<double> bufx = clState.get_buffer_mem(x);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_A = bufA.memobj;
  cl_mem Mem_x = bufx.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offA = bufA.offset;
  int offx = bufx.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_A);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Aptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_A, true, CL_MAP_READ,
        sizeof(double) * offA, sizeof(double) * M * N, 0, NULL, NULL, NULL));
    double* xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_x, true, CL_MAP_READ,
        sizeof(double) * offx,
        sizeof(double) * (TransA == CblasTrans) ? M : N, 0,
        NULL,
        NULL, NULL));
    double* yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_y, true,
        CL_MAP_READ | CL_MAP_WRITE,
        sizeof(double) * offy,
        sizeof(double) * (TransA == CblasTrans) ? N : M, 0,
        NULL,
        NULL, NULL));

    caffe_cpu_gemv<double>(TransA, M, N, alpha, Aptr, xptr, beta, yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_A, Aptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_x, xptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_y, yptr,
    0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    clblasTranspose clTransA =
    (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    GREENTEA_CL_BLAS_CHECK(
      clblasDgemv(clblasRowMajor,
            clTransA, M, N, alpha, Mem_A, offA, N, Mem_x, offx, 1,
            beta, Mem_y, offy, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    clblast::Layout layout = clblast::Layout::kRowMajor;
    clblast::Transpose a_transpose = (TransA == CblasNoTrans) ?
      clblast::Transpose::kNo : clblast::Transpose::kYes;

    const size_t ldA = N;
    const size_t incx = 1;
    const size_t incy = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Gemv<double>(
        layout, a_transpose,
        M, N,
        alpha,
        Mem_A, offA, ldA,
        Mem_x, offx, incx,
        beta,
        Mem_y, offy, incy,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(
        Mem_x, size_type((TransA == CblasTrans) ? M : N), size_type(offx),
        difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(
        Mem_y, size_type((TransA == CblasTrans) ? N : M), size_type(offy),
        difference_type(1), ctx);
    viennacl::matrix_base<double, size_t, ptrdiff_t> mat(Mem_A, ctx,
                                                         size_type(M),
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

template<>
void caffe_gpu_axpy<float>(const int_tp N, const float alpha, const float* X,
                           float* Y) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufX = clState.get_buffer_mem(X);
  ClMemOff<float> bufY = clState.get_buffer_mem(Y);

  cl_mem Mem_X = bufX.memobj;
  cl_mem Mem_Y = bufY.memobj;

  int offX = bufX.offset;
  int offY = bufY.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_X, true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * N, 0, NULL, NULL, NULL));
    float* Yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_Y, true, CL_MAP_WRITE,
        sizeof(float) * offY, sizeof(float) * N, 0, NULL, NULL, NULL));

    caffe_axpy<float>(N, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_X, Xptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_Y, Yptr,
    0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    GREENTEA_CL_BLAS_CHECK(
        clblasSaxpy(N, alpha, Mem_X, offX,
            1, Mem_Y, offY, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Axpy<float>(
        N,
        alpha,
        Mem_X, offX, incX,
        Mem_Y, offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(Mem_X, size_type(N),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(Mem_Y, size_type(N),
                                                     size_type(offY),
                                                     difference_type(1), ctx);
    v2 += alpha * v1;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template<>
void caffe_gpu_axpy<double>(const int_tp N, const double alpha, const double* X,
                            double* Y) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufX = clState.get_buffer_mem(X);
  ClMemOff<double> bufY = clState.get_buffer_mem(Y);

  cl_mem Mem_X = bufX.memobj;
  cl_mem Mem_Y = bufY.memobj;

  int offX = bufX.offset;
  int offY = bufY.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_X, true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * N, 0, NULL, NULL, NULL));
    double* Yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_Y, true, CL_MAP_WRITE,
        sizeof(double) * offY, sizeof(double) * N, 0, NULL, NULL, NULL));

    caffe_axpy<double>(N, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_X, Xptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_Y, Yptr,
    0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    GREENTEA_CL_BLAS_CHECK(
        clblasDaxpy(N, alpha, Mem_X, offX,
            1, Mem_Y, offY, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Axpy<double>(
        N,
        alpha,
        Mem_X, offX, incX,
        Mem_Y, offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(Mem_X, size_type(N),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(Mem_Y, size_type(N),
                                                     size_type(offY),
                                                     difference_type(1), ctx);
    v2 += alpha * v1;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

void caffe_gpu_memcpy(const uint_tp N, const void* X, void* Y) {
  if (X == Y) return;

  ClState& clState = Caffe::cl_state();

  ClMemOff<uint8_t> bufX = clState.get_buffer_mem(X);
  ClMemOff<uint8_t> bufY = clState.get_buffer_mem(Y);
  int dev_id;
  if (bufX.memobj != NULL) dev_id = clState.get_mem_dev(bufX.memobj);
  else
      dev_id = clState.get_mem_dev(bufY.memobj);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (bufX.memobj != NULL && bufY.memobj != NULL) {
    clEnqueueCopyBuffer(ctx.get_queue().handle().get(), bufX.memobj,
                        bufY.memobj, bufX.offset,
                        bufY.offset, N, 0, NULL, NULL);
  } else if (bufX.memobj != NULL) {
    clEnqueueReadBuffer(ctx.get_queue().handle().get(), bufX.memobj,
                        CL_TRUE, bufX.offset, N,
                        Y, 0, NULL, NULL);
  } else if (bufY.memobj != NULL) {
    clEnqueueWriteBuffer(ctx.get_queue().handle().get(), bufY.memobj,
                         CL_TRUE, bufY.offset, N,
                         X, 0, NULL, NULL);
  } else {
    memcpy(Y, X, N);
  }
}

template<>
void caffe_gpu_scal<float>(const int_tp N, const float alpha, float *X) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufX = clState.get_buffer_mem(X);

  cl_mem Mem_X = bufX.memobj;

  int offX = bufX.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_X, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(float) * offX, sizeof(float) * N, 0, NULL, NULL, NULL));

    caffe_scal<float>(N, alpha, xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_X, xptr,
    0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    GREENTEA_CL_BLAS_CHECK(clblasSscal(N, alpha, Mem_X, offX,
            1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incx = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Scal<float>(
        N,
        alpha,
        Mem_X, offX, incx,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(Mem_X, size_type(N),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    v1 *= alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template<>
void caffe_gpu_scal<double>(const int_tp N, const double alpha, double *X) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufX = clState.get_buffer_mem(X);

  cl_mem Mem_X = bufX.memobj;

  int offX = bufX.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_X, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(double) * offX, sizeof(double) * N, 0, NULL, NULL, NULL));

    caffe_scal<double>(N, alpha, xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_X, xptr,
    0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    GREENTEA_CL_BLAS_CHECK(clblasDscal(N, alpha, Mem_X, offX,
            1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incx = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Scal<double>(
        N,
        alpha,
        Mem_X, offX, incx,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(Mem_X, size_type(N),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    v1 *= alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template<>
void caffe_gpu_axpby<float>(const int_tp N, const float alpha, const float* X,
                            const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template<>
void caffe_gpu_axpby<double>(const int_tp N, const double alpha,
                             const double* X, const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template<>
void caffe_gpu_dot<float>(const int_tp n, const float* x, const float* y,
                          float* out) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufX = clState.get_buffer_mem(x);
  ClMemOff<float> bufY = clState.get_buffer_mem(y);

  cl_mem Mem_X = bufX.memobj;
  cl_mem Mem_Y = bufY.memobj;

  int offX = bufX.offset;
  int offY = bufY.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_X, true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * n, 0, NULL, NULL, NULL));
    float* Yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_Y, true, CL_MAP_READ,
        sizeof(float) * offY, sizeof(float) * n, 0, NULL, NULL, NULL));

    *out = caffe_cpu_dot<float>(n, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_X, Xptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_Y, Yptr,
    0, NULL, NULL);

  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    float* memDotP = static_cast<float*>(clState.create_buffer(dev_id,
        CL_MEM_WRITE_ONLY, sizeof(cl_float), NULL, &err));
    float* memScratch = static_cast<float*>(clState.create_buffer(dev_id,
        CL_MEM_READ_WRITE, n * sizeof(cl_float), NULL, &err));
    ClMemOff<float> bufDotP = clState.get_buffer_mem(memDotP);
    ClMemOff<float> bufScratch = clState.get_buffer_mem(memScratch);

    GREENTEA_CL_BLAS_CHECK(
          clblasSdot(n, bufDotP.memobj, 0, Mem_X, offX, 1, Mem_Y,
              offY, 1, bufScratch.memobj, 1, &queue, 0, NULL, NULL));

    caffe_gpu_memcpy(sizeof(float), memDotP, out);

    clState.destroy_buffer(memScratch);
    clState.destroy_buffer(memDotP);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;

    float* Z = static_cast<float*>(clState.create_buffer(dev_id,
        CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err));
    ClMemOff<float> bufZ = clState.get_buffer_mem(Z);
    cl_mem Mem_Z = bufZ.memobj;
    // TODO: error handling.

    const size_t offZ = 0;
    const size_t incX = 1;
    const size_t incY = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Dot<float>(
        n,
        Mem_Z, offZ,
        Mem_X, offX, incX,
        Mem_Y, offY, incY,
        &queue));

    caffe_gpu_memcpy(sizeof(float), Z, out);
    clState.destroy_buffer(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(Mem_X, size_type(n),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(Mem_Y, size_type(n),
                                                     size_type(offY),
                                                     difference_type(1), ctx);

    *out = viennacl::linalg::inner_prod(v1, v2);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template<>
void caffe_gpu_dot<double>(const int_tp n, const double* x, const double* y,
                           double * out) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufX = clState.get_buffer_mem(x);
  ClMemOff<double> bufY = clState.get_buffer_mem(y);

  cl_mem Mem_X = bufX.memobj;
  cl_mem Mem_Y = bufY.memobj;

  int offX = bufX.offset;
  int offY = bufY.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
  double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
      ctx.get_queue().handle().get(), Mem_X, true, CL_MAP_READ,
      sizeof(double) * offX, sizeof(double) * n, 0, NULL, NULL, NULL));
  double* Yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
      ctx.get_queue().handle().get(), Mem_Y, true, CL_MAP_READ,
      sizeof(double) * offY, sizeof(double) * n, 0, NULL, NULL, NULL));

  *out = caffe_cpu_dot<double>(n, Xptr, Yptr);

  clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_X, Xptr, 0, NULL,
  NULL);
  clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_Y, Yptr, 0, NULL,
  NULL);

  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    double* memDotP = static_cast<double*>(clState.create_buffer(dev_id,
        CL_MEM_WRITE_ONLY, sizeof(cl_double), NULL, &err));
    double* memScratch = static_cast<double*>(clState.create_buffer(dev_id,
        CL_MEM_READ_WRITE, n * sizeof(cl_float), NULL, &err));
    ClMemOff<double> bufDotP = clState.get_buffer_mem(memDotP);
    ClMemOff<double> bufScratch = clState.get_buffer_mem(memScratch);

    GREENTEA_CL_BLAS_CHECK(
          clblasDdot(n, bufDotP.memobj, 0, Mem_X, offX, 1, Mem_Y,
              offY, 1, bufScratch.memobj, 1, &queue, 0, NULL, NULL));

    caffe_gpu_memcpy(sizeof(double), memDotP, out);

    clState.destroy_buffer(memScratch);
    clState.destroy_buffer(memDotP);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;

    double* Z = static_cast<double*>(clState.create_buffer(dev_id,
        CL_MEM_READ_WRITE, sizeof(cl_double), NULL, &err));
    ClMemOff<double> bufZ = clState.get_buffer_mem(Z);
    cl_mem Mem_Z = bufZ.memobj;
    // TODO: error handling.

    const size_t offZ = 0;
    const size_t incX = 1;
    const size_t incY = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Dot<double>(
        n,
        Mem_Z, offZ,
        Mem_X, offX, incX,
        Mem_Y, offY, incY,
        &queue));

    caffe_gpu_memcpy(sizeof(double), Z, out);
    clState.destroy_buffer(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(Mem_X, size_type(n),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(Mem_Y, size_type(n),
                                                     size_type(offY),
                                                     difference_type(1), ctx);

    *out = viennacl::linalg::inner_prod(v1, v2);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template<>
void caffe_gpu_asum<float>(const int_tp n, const float* x, float* y) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufX = clState.get_buffer_mem(x);

  cl_mem Mem_X = bufX.memobj;

  int offX = bufX.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_X, true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * n, 0, NULL, NULL, NULL));

    *y = caffe_cpu_asum<float>(n, Xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_X, Xptr,
    0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    float* memAsum = static_cast<float*>(clState.create_buffer(dev_id,
      CL_MEM_WRITE_ONLY, sizeof(cl_float), NULL, &err));
    float* memScratch = static_cast<float*>(clState.create_buffer(dev_id,
      CL_MEM_READ_WRITE, n * sizeof(cl_float), NULL, &err));
    ClMemOff<float> bufAsum = clState.get_buffer_mem(memAsum);
    ClMemOff<float> bufScratch = clState.get_buffer_mem(memScratch);

    GREENTEA_CL_BLAS_CHECK(
        clblasSasum(n, bufAsum.memobj, 0, Mem_X, offX, 1,
            bufScratch.memobj, 1, &queue, 0, NULL, NULL));

    caffe_gpu_memcpy(sizeof(float), memAsum, y);

    clState.destroy_buffer(memScratch);
    clState.destroy_buffer(memAsum);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;
    float* Z = static_cast<float*>(clState.create_buffer(dev_id,
      CL_MEM_WRITE_ONLY, sizeof(cl_float), NULL, &err));
    ClMemOff<float> bufZ = clState.get_buffer_mem(Z);
    cl_mem Mem_Z = bufZ.memobj;
    // TODO: error handling.

    const size_t offZ = 0;
    const size_t incX = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Asum<float>(
        n,
        Mem_Z, offZ,
        Mem_X, offX, incX,
        &queue));

    caffe_gpu_memcpy(sizeof(float), Z, Y);

    clState.destroy_buffer(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
         uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(Mem_X, size_type(n),
                                                      size_type(offX),
                                                      difference_type(1), ctx);

    *y = viennacl::linalg::norm_1(v1);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template<>
void caffe_gpu_asum<double>(const int_tp n, const double* x, double* y) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufX = clState.get_buffer_mem(x);

  cl_mem Mem_X = bufX.memobj;

  int offX = bufX.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_X, true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * n, 0, NULL, NULL, NULL));

    *y = caffe_cpu_asum<double>(n, Xptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_X, Xptr,
    0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    double* memAsum = static_cast<double*>(clState.create_buffer(dev_id,
      CL_MEM_WRITE_ONLY, sizeof(cl_double), NULL, &err));
    double* memScratch = static_cast<double*>(clState.create_buffer(dev_id,
      CL_MEM_READ_WRITE, n * sizeof(cl_double), NULL, &err));
    ClMemOff<double> bufAsum = clState.get_buffer_mem(memAsum);
    ClMemOff<double> bufScratch = clState.get_buffer_mem(memScratch);

    GREENTEA_CL_BLAS_CHECK(
        clblasDasum(n, bufAsum.memobj, 0, Mem_X, offX, 1,
            bufScratch.memobj, 1, &queue, 0, NULL, NULL));

    caffe_gpu_memcpy(sizeof(double), memAsum, y);

    clState.destroy_buffer(memScratch);
    clState.destroy_buffer(memAsum);

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err = CL_SUCCESS;
    double* Z = static_cast<double*>(clState.create_buffer(dev_id,
      CL_MEM_WRITE_ONLY, sizeof(cl_double), NULL, &err));
    ClMemOff<double> bufZ = clState.get_buffer_mem(Z);
    cl_mem Mem_Z = bufZ.memobj;
    // TODO: error handling.

    const size_t offZ = 0;
    const size_t incX = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Asum<double>(
        n,
        Mem_Z, offZ,
        Mem_X, offX, incX,
        &queue));

    caffe_gpu_memcpy(sizeof(double), Z, Y);

    clState.destroy_buffer(Z);

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
         uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(Mem_X, size_type(n),
                                                       size_type(offX),
                                                       difference_type(1), ctx);

    *y = viennacl::linalg::norm_1(v1);

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template<>
void caffe_gpu_scale<float>(const int_tp n, const float alpha, const float *x,
                            float* y) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<float> bufX = clState.get_buffer_mem(x);
  ClMemOff<float> bufY = clState.get_buffer_mem(y);

  cl_mem Mem_X = bufX.memobj;
  cl_mem Mem_Y = bufY.memobj;

  int offX = bufX.offset;
  int offY = bufY.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    float* Xptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_X, true, CL_MAP_READ,
        sizeof(float) * offX, sizeof(float) * n, 0, NULL, NULL, NULL));
    float* Yptr = reinterpret_cast<float*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_Y, true, CL_MAP_WRITE,
        sizeof(float) * offY, sizeof(float) * n, 0, NULL, NULL, NULL));

    caffe_cpu_scale<float>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_X, Xptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_Y, Yptr,
    0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    // FIXME: Remove, as can reuse ctx obtained above?
    cl_command_queue queue = ctx.get_queue().handle().get();

    // FIXME: Use xAXPY with beta = 0?
    GREENTEA_CL_BLAS_CHECK(
      clblasScopy(n, Mem_X, offX, 1, Mem_Y, offY, 1, 1, &queue, 0, NULL, NULL));
    GREENTEA_CL_BLAS_CHECK(
      clblasSscal(n, alpha, Mem_Y, offY, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Copy<float>(
        n,
        Mem_X, offX, incX,
        Mem_Y, offY, incY,
        &queue));
    GREENTEA_CLBLAST_CHECK(
      clblast::Scal<float>(
        n,
        alpha,
        Mem_Y, offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<float,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<float, size_t, ptrdiff_t> v1(Mem_X, size_type(n),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<float, size_t, ptrdiff_t> v2(Mem_Y, size_type(n),
                                                     size_type(offY),
                                                     difference_type(1), ctx);

    v2 = v1 * alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template<>
void caffe_gpu_scale<double>(const int_tp n, const double alpha,
                             const double *x, double* y) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<double> bufX = clState.get_buffer_mem(x);
  ClMemOff<double> bufY = clState.get_buffer_mem(y);

  cl_mem Mem_X = bufX.memobj;
  cl_mem Mem_Y = bufY.memobj;

  int offX = bufX.offset;
  int offY = bufY.offset;

  int dev_id = clState.get_mem_dev(Mem_X);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    double* Xptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_X, true, CL_MAP_READ,
        sizeof(double) * offX, sizeof(double) * n, 0, NULL, NULL, NULL));
    double* Yptr = reinterpret_cast<double*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), Mem_Y, true, CL_MAP_WRITE,
        sizeof(double) * offY, sizeof(double) * n, 0, NULL, NULL, NULL));

    caffe_cpu_scale<double>(n, alpha, Xptr, Yptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_X, Xptr,
    0, NULL, NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), Mem_Y, Yptr,
    0, NULL, NULL);
  } else {
#if defined(USE_CLBLAS)

    // FIXME: Remove, as can reuse ctx obtained above?
    cl_command_queue queue = ctx.get_queue().handle().get();

    // FIXME: Use xAXPY with beta = 0?
    GREENTEA_CL_BLAS_CHECK(
      clblasDcopy(n, Mem_X, offX, 1, Mem_Y, offY, 1, 1, &queue, 0, NULL, NULL));
    GREENTEA_CL_BLAS_CHECK(
      clblasDscal(n, alpha, Mem_Y, offY, 1, 1, &queue, 0, NULL, NULL));

#elif defined(USE_CLBLAST)

    cl_command_queue queue = ctx.get_queue().handle().get();

    const size_t incX = 1;
    const size_t incY = 1;

    GREENTEA_CLBLAST_CHECK(
      clblast::Copy<double>(
        n,
        Mem_X, offX, incX,
        Mem_Y, offY, incY,
        &queue));
    GREENTEA_CLBLAST_CHECK(
      clblast::Scal<double>>(
        n,
        alpha,
        Mem_Y, offY, incY,
        &queue));

#else  // default (ViennaCL)

    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::vector_base<double,
        uint_tp, int_tp>::size_type difference_type;

    viennacl::vector_base<double, size_t, ptrdiff_t> v1(Mem_X, size_type(n),
                                                     size_type(offX),
                                                     difference_type(1), ctx);
    viennacl::vector_base<double, size_t, ptrdiff_t> v2(Mem_Y, size_type(n),
                                                     size_type(offY),
                                                     difference_type(1), ctx);

    v2 = v1 * alpha;

#endif  // clBLAS, CLBlast, or default (ViennaCL)
  }
}

template<typename Dtype>
void caffe_gpu_set(const int_tp N, const Dtype alpha, Dtype* Y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<Dtype> bufY = clState.get_buffer_mem(Y);

  cl_mem Mem_Y = bufY.memobj;

  int offY = bufY.offset;

  int dev_id = clState.get_mem_dev(Mem_Y);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  // OpenCL Version >= 1.2 approach
  // clEnqueueFillBuffer(ctx.get_queue().handle().get(),
  //                  Y, &alpha, sizeof(Dtype),
  //                  offY, N, 0, NULL, NULL);

  // OpenCL Version < 1.2 fallback
  viennacl::ocl::kernel &oclk_fill = program.get_kernel(
      CL_KERNEL_SELECT("fill"));
  viennacl::ocl::enqueue(oclk_fill(N, alpha, WrapHandle(Mem_Y, &ctx), offY),
                         ctx.get_queue());
}

template void caffe_gpu_set<int_tp>(const int_tp N, const int_tp alpha,
                                    int_tp* Y);
template void caffe_gpu_set<float>(const int_tp N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int_tp N, const double alpha,
                                    double* Y);

template<typename Dtype>
void caffe_gpu_sign(const int_tp n, const Dtype* x, Dtype* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<Dtype> bufx= clState.get_buffer_mem(x);
  ClMemOff<Dtype> bufy= clState.get_buffer_mem(y);

  cl_mem Mem_x = bufx.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offx = bufx.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_x);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sign = program.get_kernel(
      CL_KERNEL_SELECT("sign"));
  viennacl::ocl::enqueue(
      oclk_sign(n, WrapHandle(Mem_x, &ctx), offx,
                WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template void caffe_gpu_sign<float>(const int_tp n,
                                    const float* x,
                                    float* y);
template void caffe_gpu_sign<double>(const int_tp n,
                                     const double* x,
                                     double* y);

template<typename Dtype>
void caffe_gpu_sgnbit(const int_tp n, const Dtype* x, Dtype* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<Dtype> bufx= clState.get_buffer_mem(x);
  ClMemOff<Dtype> bufy= clState.get_buffer_mem(y);

  cl_mem Mem_x = bufx.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offx = bufx.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_x);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sgnbit = program.get_kernel(
      CL_KERNEL_SELECT("sgnbit"));
  viennacl::ocl::enqueue(
      oclk_sgnbit(n, WrapHandle(Mem_x, &ctx), offx,
                  WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template void caffe_gpu_sgnbit<float>(const int_tp n,
                                      const float* x,
                                      float* y);
template void caffe_gpu_sgnbit<double>(const int_tp n,
                                       const double* x,
                                       double* y);

template<>
void caffe_gpu_add_scalar(const int_tp N, const float alpha, float* Y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<float> bufY = clState.get_buffer_mem(Y);

  cl_mem Mem_Y = bufY.memobj;

  int offY = bufY.offset;

  int dev_id = clState.get_mem_dev(Mem_Y);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();
  viennacl::ocl::kernel &oclk_add_scalar = program.get_kernel(
      // CL_KERNEL_SELECT("add_scalar"));
      "add_scalar" "_float");
  viennacl::ocl::enqueue(oclk_add_scalar(N, alpha,
                         WrapHandle(Mem_Y, &ctx), offY),
                         ctx.get_queue());
}

template<>
void caffe_gpu_add_scalar(const int_tp N, const double alpha, double* Y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<double> bufY = clState.get_buffer_mem(Y);

  cl_mem Mem_Y = bufY.memobj;

  int offY = bufY.offset;

  int dev_id = clState.get_mem_dev(Mem_Y);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();
  viennacl::ocl::kernel &oclk_add_scalar = program.get_kernel(
      // CL_KERNEL_SELECT("add_scalar"));
      "add_scalar" "_double");
  viennacl::ocl::enqueue(oclk_add_scalar(N, alpha,
                         WrapHandle(Mem_Y, &ctx), offY),
                         ctx.get_queue());
}

template<>
void caffe_gpu_add<float>(const int_tp N, const float* a, const float* b,
                          float* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<float> bufa = clState.get_buffer_mem(a);
  ClMemOff<float> bufb = clState.get_buffer_mem(b);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_b = bufb.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offb = bufb.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_add = program.get_kernel("add" "_float");
  viennacl::ocl::enqueue(
      oclk_add(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_b, &ctx), offb,
               WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_add<double>(const int_tp N, const double* a, const double* b,
                           double* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<double> bufa = clState.get_buffer_mem(a);
  ClMemOff<double> bufb = clState.get_buffer_mem(b);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_b = bufb.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offb = bufb.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_add = program.get_kernel("add" "_double");
  viennacl::ocl::enqueue(
      oclk_add(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_b, &ctx), offb,
               WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_sub<float>(const int_tp N, const float* a, const float* b,
                          float* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<float> bufa = clState.get_buffer_mem(a);
  ClMemOff<float> bufb = clState.get_buffer_mem(b);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_b = bufb.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offb = bufb.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sub = program.get_kernel("sub" "_float");
  viennacl::ocl::enqueue(
      oclk_sub(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_b, &ctx), offb,
               WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_sub<double>(const int_tp N, const double* a, const double* b,
                           double* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<double> bufa = clState.get_buffer_mem(a);
  ClMemOff<double> bufb = clState.get_buffer_mem(b);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_b = bufb.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offb = bufb.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_sub = program.get_kernel("sub" "_double");
  viennacl::ocl::enqueue(
      oclk_sub(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_b, &ctx), offb,
               WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_mul<float>(const int_tp N, const float* a, const float* b,
                          float* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<float> bufa = clState.get_buffer_mem(a);
  ClMemOff<float> bufb = clState.get_buffer_mem(b);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_b = bufb.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offb = bufb.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_mul = program.get_kernel("mul" "_float");
  viennacl::ocl::enqueue(
      oclk_mul(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_b, &ctx), offb,
               WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_mul<double>(const int_tp N, const double* a, const double* b,
                           double* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<double> bufa = clState.get_buffer_mem(a);
  ClMemOff<double> bufb = clState.get_buffer_mem(b);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_b = bufb.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offb = bufb.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_mul = program.get_kernel("mul" "_double");
  viennacl::ocl::enqueue(
      oclk_mul(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_b, &ctx), offb,
               WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_div<float>(const int_tp N, const float* a, const float* b,
                          float* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<float> bufa = clState.get_buffer_mem(a);
  ClMemOff<float> bufb = clState.get_buffer_mem(b);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_b = bufb.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offb = bufb.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_div = program.get_kernel("div" "_float");
  viennacl::ocl::enqueue(
      oclk_div(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_b, &ctx), offb,
               WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_div<double>(const int_tp N, const double* a, const double* b,
                           double* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<double> bufa = clState.get_buffer_mem(a);
  ClMemOff<double> bufb = clState.get_buffer_mem(b);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_b = bufb.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offb = bufb.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_div = program.get_kernel("div" "_double");
  viennacl::ocl::enqueue(
      oclk_div(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_b, &ctx), offb,
               WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_abs<float>(const int_tp N, const float* a, float* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<float> bufa = clState.get_buffer_mem(a);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_abs = program.get_kernel("abs" "_float");
  viennacl::ocl::enqueue(
      oclk_abs(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_abs<double>(const int_tp N, const double* a, double* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<double> bufa = clState.get_buffer_mem(a);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_abs = program.get_kernel("abs" "_double");
  viennacl::ocl::enqueue(
      oclk_abs(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_exp<float>(const int_tp N, const float* a, float* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<float> bufa = clState.get_buffer_mem(a);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_exp = program.get_kernel("exp" "_float");
  viennacl::ocl::enqueue(
      oclk_exp(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_exp<double>(const int_tp N, const double* a, double* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<double> bufa = clState.get_buffer_mem(a);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_exp = program.get_kernel("exp" "_double");
  viennacl::ocl::enqueue(
      oclk_exp(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_log<float>(const int_tp N, const float* a, float* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<float> bufa = clState.get_buffer_mem(a);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_log = program.get_kernel("log" "_float");
  viennacl::ocl::enqueue(
      oclk_log(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_log<double>(const int_tp N, const double* a, double* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<double> bufa = clState.get_buffer_mem(a);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_log = program.get_kernel("log" "double");
  viennacl::ocl::enqueue(
      oclk_log(N, WrapHandle(Mem_a, &ctx), offa, WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_powx<float>(const int_tp N, const float* a, const float alpha,
                           float* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<float> bufa = clState.get_buffer_mem(a);
  ClMemOff<float> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_powx = program.get_kernel(
      "powx" "_float");
  viennacl::ocl::enqueue(
      oclk_powx(N, WrapHandle(Mem_a, &ctx), offa, alpha,
                WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

template<>
void caffe_gpu_powx<double>(const int_tp N, const double* a, const double alpha,
                            double* y) {
  ClState& clState = Caffe::cl_state();

  ClMemOff<double> bufa = clState.get_buffer_mem(a);
  ClMemOff<double> bufy = clState.get_buffer_mem(y);

  cl_mem Mem_a = bufa.memobj;
  cl_mem Mem_y = bufy.memobj;

  int offa = bufa.offset;
  int offy = bufy.offset;

  int dev_id = clState.get_mem_dev(Mem_a);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &program = (Caffe::Get().GetDevice(dev_id, false))
      ->program();

  viennacl::ocl::kernel &oclk_powx = program.get_kernel(
      "powx" "_double");
  viennacl::ocl::enqueue(
      oclk_powx(N, WrapHandle(Mem_a, &ctx), offa, alpha,
                WrapHandle(Mem_y, &ctx), offy),
      ctx.get_queue());
}

void caffe_gpu_rng_uniform(const int_tp n, unsigned int* r) {
  std::vector<uint_tp> random(n);  //NOLINT
  caffe_rng_uniform(n, &random[0]);
  caffe_gpu_memcpy(sizeof(uint_tp) * n, &random[0], r);
}

template<>
void caffe_gpu_rng_uniform<float>(const int_tp n, const float a, const float b,
                                  float* r) {
  std::vector<float> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  caffe_gpu_memcpy(sizeof(float) * n, &random[0], r);
}

template<>
void caffe_gpu_rng_uniform<double>(const int_tp n, const double a,
                                   const double b, double* r) {
  std::vector<double> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  caffe_gpu_memcpy(sizeof(double) * n, &random[0], r);
}

template<>
void caffe_gpu_rng_gaussian(const int_tp n, const float mu, const float sigma,
                            float* r) {
  std::vector<float> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  caffe_gpu_memcpy(sizeof(float) * n, &random[0], r);
}

template<>
void caffe_gpu_rng_gaussian(const int_tp n, const double mu, const double sigma,
                            double* r) {
  std::vector<double> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  caffe_gpu_memcpy(sizeof(double) * n, &random[0], r);
}
}  // namespace caffe
#endif  // USE_GREENTEA
