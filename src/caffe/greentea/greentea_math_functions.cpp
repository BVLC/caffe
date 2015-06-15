/*
 * greentea_math_functions.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "caffe/common.hpp"

#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"

#include "caffe/util/math_functions.hpp"

#include <sys/time.h>
#include <random>
#include <functional>

#ifdef USE_CLBLAS
#include <clBLAS.h>
#endif

#ifdef USE_VIENNACLBLAS
#include "libviennacl/include/viennacl.hpp"
#endif

namespace caffe {

void greentea_memset(const int ctx_id, const size_t N, const int alpha,
                     cl_mem X, const int offX) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);
  // OpenCL Version >= 1.2 approach
  // clEnqueueFillBuffer(ctx.get_queue().handle().get(), X, &alpha, sizeof(int),
  //                     offX, N, 0, NULL, NULL);
  // OpenCL Version < 1.2 fallback
  typedef float Dtype;
  viennacl::ocl::kernel &oclk_fill = program.get_kernel(
      CL_KERNEL_SELECT("fill"));
  viennacl::ocl::enqueue(
      oclk_fill(int(N / sizeof(Dtype)), Dtype(alpha), WrapHandle(X, ctx), offX),
      ctx.get_queue());
  ctx.get_queue().finish();
}

// Copy from OpenCL buffer to main memory
void greentea_gpu_memcpy(const size_t N, const cl_mem X, const int offX,
                         void *Y, viennacl::ocl::context &ctx) {
  if (Y != NULL) {
    clEnqueueReadBuffer(ctx.get_queue().handle().get(), X, CL_TRUE, offX, N, Y,
                        0,
                        NULL,
                        NULL);
  }
  ctx.get_queue().finish();
}

// Copy from main memory to OpenCL buffer
void greentea_gpu_memcpy(const size_t N, const void* X, cl_mem Y,
                         const int offY, viennacl::ocl::context &ctx) {
  if (X != NULL) {
    clEnqueueWriteBuffer(ctx.get_queue().handle().get(), Y,
    CL_TRUE,
                         offY, N, X, 0, NULL, NULL);
  }
  ctx.get_queue().finish();
}

// Copy from OpenCL to OpenCL buffer
void greentea_gpu_memcpy(const size_t N, const cl_mem X, const int offX,
                         cl_mem Y, const int offY,
                         viennacl::ocl::context &ctx) {
  clEnqueueCopyBuffer(ctx.get_queue().handle().get(), X, Y, offX, offY, N, 0,
  NULL,
                      NULL);
  ctx.get_queue().finish();
}

template<typename Dtype>
void greentea_copy(const int N, const cl_mem X, const int offX, Dtype* Y,
                   viennacl::ocl::context &ctx) {
    greentea_gpu_memcpy(sizeof(Dtype) * N, X, offX, Y, ctx);
  ctx.get_queue().finish();
}

template<typename Dtype>
void greentea_copy(const int N, const Dtype* X, cl_mem Y, const int offY,
                   viennacl::ocl::context &ctx) {
    greentea_gpu_memcpy(sizeof(Dtype) * N, X, Y, offY, ctx);
  ctx.get_queue().finish();
}

// Copy from OpenCL buffer to OpenCL buffer
template<typename Dtype>
void greentea_copy(const int N, const cl_mem X, const int offX, cl_mem Y,
                   const int offY, viennacl::ocl::context &ctx) {
  if (X != Y) {
    greentea_gpu_memcpy(sizeof(Dtype) * N, X, offX, Y, offY, ctx);
  }
  ctx.get_queue().finish();
}

// Explicit instantiations
template void greentea_copy<int>(const int N, const cl_mem X, const int offX,
                                 int* Y, viennacl::ocl::context &ctx);
template void greentea_copy<unsigned int>(const int N, const cl_mem X,
                                          const int offX, unsigned int* Y,
                                          viennacl::ocl::context &ctx);
template void greentea_copy<float>(const int N, const cl_mem X, const int offX,
                                   float* Y, viennacl::ocl::context &ctx);
template void greentea_copy<double>(const int N, const cl_mem X, const int offX,
                                    double* Y, viennacl::ocl::context &ctx);
template void greentea_copy<int>(const int N, const int* X, cl_mem Y,
                                 const int offY, viennacl::ocl::context &ctx);
template void greentea_copy<unsigned int>(const int N, const unsigned int* X,
                                          cl_mem Y, const int offY,
                                          viennacl::ocl::context &ctx);
template void greentea_copy<float>(const int N, const float* X, cl_mem Y,
                                   const int offY, viennacl::ocl::context &ctx);
template void greentea_copy<double>(const int N, const double* X, cl_mem Y,
                                    const int offY,
                                    viennacl::ocl::context &ctx);
template void greentea_copy<int>(const int N, const cl_mem X, const int offX,
                                 cl_mem Y, const int offY,
                                 viennacl::ocl::context &ctx);
template void greentea_copy<unsigned int>(const int N, const cl_mem X,
                                          const int offX, cl_mem Y,
                                          const int offY,
                                          viennacl::ocl::context &ctx);
template void greentea_copy<float>(const int N, const cl_mem X, const int offX,
                                   cl_mem Y, const int offY,
                                   viennacl::ocl::context &ctx);
template void greentea_copy<double>(const int N, const cl_mem X, const int offX,
                                    cl_mem Y, const int offY,
                                    viennacl::ocl::context &ctx);

template<typename Dtype>
void greentea_gpu_gemm(const int ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int M, const int N,
                       const int K, const Dtype alpha, const cl_mem A,
                       const int offA, const cl_mem B, const int offB,
                       const Dtype beta, cl_mem C, const int offC) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    // Make sure the OpenCL queue is empty before using CBLAS
    ctx.get_queue().finish();
    Dtype *Aptr, *Bptr, *Cptr;
    clGetMemObjectInfo(A, CL_MEM_HOST_PTR, sizeof(Dtype*), &Aptr, NULL);
    clGetMemObjectInfo(B, CL_MEM_HOST_PTR, sizeof(Dtype*), &Bptr, NULL);
    clGetMemObjectInfo(C, CL_MEM_HOST_PTR, sizeof(Dtype*), &Cptr, NULL);

    caffe_cpu_gemm<Dtype>(TransA, TransB, M, N, K, alpha, Aptr + offA,
                          Bptr + offB, beta, Cptr + offC);
  } else {

    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    int ldc = N;

#ifdef USE_VIENNACLBLAS

    ViennaCLBackend backend;
    ViennaCLBackendCreate(&backend);
    ViennaCLBackendSetOpenCLContextID(backend,
                                      static_cast<ViennaCLInt>(ctx_id));

    ViennaCLTranspose vclTransA =
        (TransA == CblasNoTrans) ? ViennaCLNoTrans : ViennaCLTrans;
    ViennaCLTranspose vclTransB =
        (TransB == CblasNoTrans) ? ViennaCLNoTrans : ViennaCLTrans;

    ViennaCLOrder vclOrderA = ViennaCLRowMajor;
    ViennaCLOrder vclOrderB = ViennaCLRowMajor;
    ViennaCLOrder vclOrderC = ViennaCLRowMajor;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLSgemm(backend, vclOrderA, vclTransA, vclOrderB,
                              vclTransB, vclOrderC, M, N, K, alpha, A, 0, offA,
                              1, 1, lda, B, 0, offB, 1, 1, ldb, beta, C, 0,
                              offC, 1, 1, ldc));
    } else {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLDgemm(backend, vclOrderA, vclTransA, vclOrderB,
                              vclTransB, vclOrderC, M, N, K, alpha, A, 0, offA,
                              1, 1, lda, B, 0, offB, 1, 1, ldb, beta, C, 0,
                              offC, 1, 1, ldc));
    }
#endif
#ifdef USE_CLBLAS
    clblasOrder clOrder = clblasRowMajor;
    clblasTranspose clTransA =
        (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
    clblasTranspose clTransB =
        (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSgemm(clOrder, clTransA, clTransB, M, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDgemm(clOrder, clTransA, clTransB, M, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc, 1, &queue, 0, NULL, NULL));
    }
#endif
  }
}

template void greentea_gpu_gemm<float>(const int ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int M, const int N, const int K,
                                       const float alpha, const cl_mem A,
                                       const int offA, const cl_mem B,
                                       const int offB, const float beta,
                                       cl_mem C, const int offC);
template void greentea_gpu_gemm<double>(const int ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const CBLAS_TRANSPOSE TransB,
                                        const int M, const int N, const int K,
                                        const double alpha, const cl_mem A,
                                        const int offA, const cl_mem B,
                                        const int offB, const double beta,
                                        cl_mem C, const int offC);

template<typename Dtype>
void greentea_gpu_gemv(const int ctx_id, const CBLAS_TRANSPOSE TransA,
                       const int M, const int N, const Dtype alpha,
                       const cl_mem A, const int offA, const cl_mem x,
                       const int offx, const Dtype beta, cl_mem y,
                       const int offy) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    // Make sure the OpenCL queue is empty before using CBLAS
    ctx.get_queue().finish();
    Dtype *Aptr, *xptr, *yptr;
    clGetMemObjectInfo(A, CL_MEM_HOST_PTR, sizeof(Dtype*), &Aptr, NULL);
    clGetMemObjectInfo(x, CL_MEM_HOST_PTR, sizeof(Dtype*), &xptr, NULL);
    clGetMemObjectInfo(y, CL_MEM_HOST_PTR, sizeof(Dtype*), &yptr, NULL);

    caffe_cpu_gemv<Dtype>(TransA, M, N, alpha, Aptr + offA, xptr + offx, beta,
                          yptr + offy);
  } else {

    int lda = (TransA == CblasNoTrans) ? N : M;

#ifdef USE_VIENNACLBLAS
    ViennaCLBackend backend;
    ViennaCLBackendCreate(&backend);
    ViennaCLBackendSetOpenCLContextID(backend,
                                      static_cast<ViennaCLInt>(ctx_id));

    ViennaCLOrder vclOrder = ViennaCLRowMajor;
    ViennaCLTranspose vclTransA =
        (TransA == CblasNoTrans) ? ViennaCLNoTrans : ViennaCLTrans;

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLSgemv(backend, vclOrder, vclTransA, M, N, alpha, A,
                              offA, 0, 1, 1, lda, x, offx, 1, beta, y, offy, 1));
    } else {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLDgemv(backend, vclOrder, vclTransA, M, N, alpha, A,
                              offA, 0, 1, 1, lda, x, offx, 1, beta, y, offy, 1));
    }
#endif

#ifdef USE_CLBLAS

    clblasOrder clOrder = clblasRowMajor;
    clblasTranspose clTransA =
        (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSgemv(clOrder,clTransA,M,N,alpha,A,offA,N,x,offx,1,beta,y,offy,1,1,&queue,0,NULL,NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDgemv(clOrder,clTransA,M,N,alpha,A,offA,N,x,offx,1,beta,y,offy,1,1,&queue,0,NULL,NULL));
    }
#endif
  }
}

template void greentea_gpu_gemv<float>(const int ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const int M, const int N,
                                       const float alpha, const cl_mem A,
                                       const int offA, const cl_mem x,
                                       const int offx, const float beta,
                                       cl_mem y, const int offy);
template void greentea_gpu_gemv<double>(const int ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const int M, const int N,
                                        const double alpha, const cl_mem A,
                                        const int offA, const cl_mem x,
                                        const int offx, const double beta,
                                        cl_mem y, const int offy);

template<typename Dtype>
void greentea_gpu_axpy(const int ctx_id, const int N, const Dtype alpha,
                       const cl_mem X, const int offX, cl_mem Y,
                       const int offY) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    // Make sure the OpenCL queue is empty before using CBLAS
    ctx.get_queue().finish();
    Dtype *Xptr, *Yptr;
    clGetMemObjectInfo(X, CL_MEM_HOST_PTR, sizeof(Dtype*), &Xptr, NULL);
    clGetMemObjectInfo(Y, CL_MEM_HOST_PTR, sizeof(Dtype*), &Yptr, NULL);

    caffe_axpy<Dtype>(N, alpha, Xptr + offX, Yptr + offY);
  } else {

#ifdef USE_VIENNACLBLAS
    ViennaCLBackend backend;
    ViennaCLBackendCreate(&backend);
    ViennaCLBackendSetOpenCLContextID(backend,
                                      static_cast<ViennaCLInt>(ctx_id));

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLSaxpy(backend, N, alpha, X, offX, 1, Y, offY, 1));
    } else {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLDaxpy(backend, N, alpha, X, offX, 1, Y, offY, 1));
    }
#endif

#ifdef USE_CLBLAS
    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSaxpy(N,alpha,X,offX,1,Y,offY,1,1,&queue,0,NULL,NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDaxpy(N,alpha,X,offX,1,Y,offY,1,1,&queue,0,NULL,NULL));

    }
#endif
  }
}

template void greentea_gpu_axpy<float>(const int ctx_id, const int N,
                                       const float alpha, const cl_mem X,
                                       const int offX, cl_mem Y,
                                       const int offY);
template void greentea_gpu_axpy<double>(const int ctx_id, const int N,
                                        const double alpha, const cl_mem X,
                                        const int offX, cl_mem Y,
                                        const int offY);

template<typename Dtype>
void greentea_gpu_mul(const int ctx_id, const int N, const cl_mem a,
                      const int offa, const cl_mem b, const int offb, cl_mem y,
                      const int offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_mul = program.get_kernel(CL_KERNEL_SELECT("mul"));
  viennacl::ocl::enqueue(
      oclk_mul(N, WrapHandle(a, ctx), offa, WrapHandle(b, ctx), offb,
               WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_mul<float>(const int ctx_id, const int N,
                                      const cl_mem a, const int offa,
                                      const cl_mem b, const int offb, cl_mem y,
                                      const int offy);
template void greentea_gpu_mul<double>(const int ctx_id, const int N,
                                       const cl_mem a, const int offa,
                                       const cl_mem b, const int offb, cl_mem y,
                                       const int offy);

template<typename Dtype>
void greentea_gpu_div(const int ctx_id, const int N, const cl_mem a,
                      const int offa, const cl_mem b, const int offb, cl_mem y,
                      const int offy) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_div = program.get_kernel(CL_KERNEL_SELECT("div"));
  viennacl::ocl::enqueue(
      oclk_div(N, WrapHandle(a, ctx), offa, WrapHandle(b, ctx), offb,
               WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_div<float>(const int ctx_id, const int N,
                                      const cl_mem a, const int offa,
                                      const cl_mem b, const int offb, cl_mem y,
                                      const int offy);
template void greentea_gpu_div<double>(const int ctx_id, const int N,
                                       const cl_mem a, const int offa,
                                       const cl_mem b, const int offb, cl_mem y,
                                       const int offy);

template<typename Dtype>
void greentea_gpu_scal(const int ctx_id, const int N, const Dtype alpha,
                       cl_mem x, int offx) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    // Make sure the OpenCL queue is empty before using CBLAS
    ctx.get_queue().finish();
    Dtype *xptr;
    clGetMemObjectInfo(x, CL_MEM_HOST_PTR, sizeof(Dtype*), &xptr, NULL);
    caffe_scal<Dtype>(N, alpha, xptr + offx);

  } else {
#ifdef USE_VIENNACLBLAS
    ViennaCLBackend backend;
    ViennaCLBackendCreate(&backend);
    ViennaCLBackendSetOpenCLContextID(backend,
                                      static_cast<ViennaCLInt>(ctx_id));

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLSscal(backend, N, alpha, x, offx, 1));
    } else {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLDscal(backend, N, alpha, x, offx, 1));
    }
#endif

#ifdef USE_CLBLAS
    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(clblasSscal(N,alpha,x,offx,1,1,&queue,0,NULL,NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(clblasDscal(N,alpha,x,offx,1,1,&queue,0,NULL,NULL));
    }
#endif
  }
}

template void greentea_gpu_scal<float>(const int ctx_id, const int N,
                                       const float alpha, cl_mem x,
                                       const int offx);
template void greentea_gpu_scal<double>(const int ctx_id, const int N,
                                        const double alpha, cl_mem x,
                                        const int offx);

template<typename Dtype>
void greentea_gpu_axpby(const int ctx_id, const int N, const Dtype alpha,
                        const cl_mem X, const int offX, const Dtype beta,
                        cl_mem Y, const int offY) {
  greentea_gpu_scal<Dtype>(ctx_id, N, beta, Y, offY);
  greentea_gpu_axpy<Dtype>(ctx_id, N, alpha, X, offX, Y, offY);
}

template void greentea_gpu_axpby<float>(const int ctx_id, const int N,
                                        const float alpha, const cl_mem X,
                                        const int offX, const float beta,
                                        cl_mem Y, const int offY);

template void greentea_gpu_axpby<double>(const int ctx_id, const int N,
                                         const double alpha, const cl_mem X,
                                         const int offX, const double beta,
                                         cl_mem Y, const int offY);

template<typename Dtype>
void greentea_gpu_dot(const int ctx_id, const int n, const cl_mem X,
                      const int offX, const cl_mem Y, const int offY,
                      Dtype* out) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    // Make sure the OpenCL queue is empty before using CBLAS
    ctx.get_queue().finish();
    Dtype *Xptr, *Yptr;
    clGetMemObjectInfo(X, CL_MEM_HOST_PTR, sizeof(Dtype*), &Xptr, NULL);
    clGetMemObjectInfo(Y, CL_MEM_HOST_PTR, sizeof(Dtype*), &Yptr, NULL);

    *out = caffe_cpu_dot<Dtype>(n, Xptr + offX, Yptr + offY);
  } else {

#ifdef USE_VIENNACLBLAS
    ViennaCLBackend backend;
    ViennaCLBackendCreate(&backend);
    ViennaCLBackendSetOpenCLContextID(backend,
                                      static_cast<ViennaCLInt>(ctx_id));

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLSdot(backend, n, out, X, offX, 1, Y, offY, 1));
    } else {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLDdot(backend, n, out, X, offX, 1, Y, offY, 1));
    }
#endif

#ifdef USE_CLBLAS
    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
                                   sizeof(Dtype), NULL, &err);
    cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
                                    n * sizeof(Dtype), NULL, &err);

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSdot(n,gpuout,0,X,offX,1,Y,offY,1,scratch,1,&queue,0,NULL,NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDdot(n,gpuout,0,X,offX,1,Y,offY,1,scratch,1,&queue,0,NULL,NULL));
    }

    greentea_gpu_memcpy(sizeof(Dtype), gpuout, 0, out, ctx);

    ctx.get_queue().finish();
    clReleaseMemObject(gpuout);
    clReleaseMemObject(scratch);

#endif
  }
}

template void greentea_gpu_dot<float>(const int ctx_id, const int n,
                                      const cl_mem X, const int offX,
                                      const cl_mem Y, const int offY,
                                      float* out);
template void greentea_gpu_dot<double>(const int ctx_id, const int n,
                                       const cl_mem X, const int offX,
                                       const cl_mem Y, const int offY,
                                       double* out);

template<typename Dtype>
void greentea_gpu_asum(const int ctx_id, const int n, const cl_mem X,
                       const int offX, Dtype* Y) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    // Make sure the OpenCL queue is empty before using CBLAS
    ctx.get_queue().finish();
    Dtype* Xptr;
    clGetMemObjectInfo(X, CL_MEM_HOST_PTR, sizeof(Dtype*), &Xptr, NULL);
    *Y = caffe_cpu_asum<Dtype>(n, Xptr + offX);

  } else {
#ifdef USE_VIENNACLBLAS
    ViennaCLBackend backend;
    ViennaCLBackendCreate(&backend);
    ViennaCLBackendSetOpenCLContextID(backend,
                                      static_cast<ViennaCLInt>(ctx_id));

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_VCL_BLAS_CHECK(ViennaCLOpenCLSasum(backend, n, Y, X, offX, 1));
    } else {
      GREENTEA_VCL_BLAS_CHECK(ViennaCLOpenCLDasum(backend, n, Y, X, offX, 1));
    }
#endif

#ifdef USE_CLBLAS
    cl_command_queue queue = ctx.get_queue().handle().get();

    cl_int err;
    cl_mem gpuout = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
                                   sizeof(Dtype), NULL, &err);
    cl_mem scratch = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
                                    n * sizeof(Dtype), NULL, &err);

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSasum(n,gpuout,0,X,offX,1,scratch,1,&queue,0,NULL,NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDasum(n,gpuout,0,X,offX,1,scratch,1,&queue,0,NULL,NULL));
    }

    greentea_gpu_memcpy(sizeof(Dtype), gpuout, 0, Y, ctx);

    ctx.get_queue().finish();
    clReleaseMemObject(gpuout);
    clReleaseMemObject(scratch);
#endif
  }
}

template void greentea_gpu_asum<float>(const int ctx_id, const int n,
                                       const cl_mem X, const int offX,
                                       float* Y);
template void greentea_gpu_asum<double>(const int ctx_id, const int n,
                                        const cl_mem X, const int offX,
                                        double* Y);

template<typename Dtype>
void greentea_gpu_scale(const int ctx_id, const int n, const Dtype alpha,
                        const cl_mem X, const int offX, cl_mem Y,
                        const int offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    // Make sure the OpenCL queue is empty before using CBLAS
    ctx.get_queue().finish();
    Dtype *Xptr, *Yptr;
    clGetMemObjectInfo(X, CL_MEM_HOST_PTR, sizeof(Dtype*), &Xptr, NULL);
    clGetMemObjectInfo(Y, CL_MEM_HOST_PTR, sizeof(Dtype*), &Yptr, NULL);
    caffe_cpu_scale<Dtype>(n, alpha, Xptr + offX, Yptr + offY);

  } else {
#ifdef USE_VIENNACLBLAS
    ViennaCLBackend backend;
    ViennaCLBackendCreate(&backend);
    ViennaCLBackendSetOpenCLContextID(backend,
                                      static_cast<ViennaCLInt>(ctx_id));

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLScopy(backend, n, X, offX, 1, Y, offY, 1));
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLSscal(backend, n, alpha, Y, offY, 1));
    } else {
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLDcopy(backend, n, X, offX, 1, Y, offY, 1));
      GREENTEA_VCL_BLAS_CHECK(
          ViennaCLOpenCLDscal(backend, n, alpha, Y, offY, 1));
    }
#endif

#ifdef USE_CLBLAS
    viennacl::ocl::context ctx = viennacl::ocl::get_context(ctx_id);
    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasScopy(n,X,offX,1,Y,offY,1,1,&queue,0,NULL,NULL));
      GREENTEA_CL_BLAS_CHECK(clblasSscal(n,alpha,Y,offY,1,1,&queue,0,NULL,NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDcopy(n,X,offX,1,Y,offY,1,1,&queue,0,NULL,NULL));
      GREENTEA_CL_BLAS_CHECK(clblasDscal(n,alpha,Y,offY,1,1,&queue,0,NULL,NULL));
    }
#endif
  }
}

template void greentea_gpu_scale<float>(const int ctx_id, const int n,
                                        const float alpha, const cl_mem X,
                                        const int offX, cl_mem Y,
                                        const int offY);

template void greentea_gpu_scale<double>(const int ctx_id, const int n,
                                         const double alpha, const cl_mem X,
                                         const int offX, cl_mem Y,
                                         const int offY);

template<typename Dtype>
void greentea_gpu_set(const int ctx_id, const int N, const Dtype alpha,
                      cl_mem Y, const int offY) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);
  // OpenCL Version >= 1.2 approach
  //clEnqueueFillBuffer(ctx.get_queue().handle().get(), Y, &alpha, sizeof(Dtype),
  //                     offY, N, 0, NULL, NULL);

  // OpenCL Version < 1.2 fallback
  viennacl::ocl::kernel &oclk_fill = program.get_kernel(
      CL_KERNEL_SELECT("fill"));
  viennacl::ocl::enqueue(oclk_fill(N, alpha, WrapHandle(Y, ctx), offY),
                         ctx.get_queue());

  ctx.get_queue().finish();
}

template void greentea_gpu_set<int>(const int ctx_id, const int N,
                                    const int alpha, cl_mem Y, const int offY);
template void greentea_gpu_set<float>(const int ctx_id, const int N,
                                      const float alpha, cl_mem Y,
                                      const int offY);
template void greentea_gpu_set<double>(const int ctx_id, const int N,
                                       const double alpha, cl_mem Y,
                                       const int offY);

template<typename Dtype>
void greentea_gpu_add_scalar(const int ctx_id, const int N, const Dtype alpha,
                             cl_mem Y, const int offY) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_add_scalar = program.get_kernel(
      CL_KERNEL_SELECT("add_scalar"));
  viennacl::ocl::enqueue(oclk_add_scalar(N, alpha, WrapHandle(Y, ctx), offY),
                         ctx.get_queue());
}

template void greentea_gpu_add_scalar<float>(const int ctx_id, const int N,
                                             const float alpha, cl_mem Y,
                                             const int offY);
template void greentea_gpu_add_scalar<double>(const int ctx_id, const int N,
                                              const double alpha, cl_mem Y,
                                              const int offY);

template<typename Dtype>
void greentea_gpu_add(const int ctx_id, const int n, const cl_mem a,
                      const int offa, const cl_mem b, const int offb, cl_mem y,
                      const int offy) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_add = program.get_kernel(CL_KERNEL_SELECT("add"));
  viennacl::ocl::enqueue(
      oclk_add(n, WrapHandle(a, ctx), offa, WrapHandle(b, ctx), offb,
               WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_add<float>(const int ctx_id, const int n,
                                      const cl_mem a, const int offa,
                                      const cl_mem b, const int offb, cl_mem y,
                                      const int offy);
template void greentea_gpu_add<double>(const int ctx_id, const int n,
                                       const cl_mem a, const int offa,
                                       const cl_mem b, const int offb, cl_mem y,
                                       const int offy);

template<typename Dtype>
void greentea_gpu_sub(const int ctx_id, const int n, const cl_mem a,
                      const int offa, const cl_mem b, const int offb, cl_mem y,
                      const int offy) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_sub = program.get_kernel(CL_KERNEL_SELECT("sub"));
  viennacl::ocl::enqueue(
      oclk_sub(n, WrapHandle(a, ctx), offa, WrapHandle(b, ctx), offb,
               WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_sub<float>(const int ctx_id, const int n,
                                      const cl_mem a, const int offa,
                                      const cl_mem b, const int offb, cl_mem y,
                                      const int offy);
template void greentea_gpu_sub<double>(const int ctx_id, const int n,
                                       const cl_mem a, const int offa,
                                       const cl_mem b, const int offb, cl_mem y,
                                       const int offy);

template<typename Dtype>
void greentea_gpu_abs(const int ctx_id, const int N, const cl_mem a,
                      const int offa, cl_mem y, const int offy) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_abs = program.get_kernel(CL_KERNEL_SELECT("abs"));
  viennacl::ocl::enqueue(
      oclk_abs(N, WrapHandle(a, ctx), offa, WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_abs<float>(const int ctx_id, const int N,
                                      const cl_mem a, const int offa, cl_mem y,
                                      const int offy);
template void greentea_gpu_abs<double>(const int ctx_id, const int N,
                                       const cl_mem a, const int offa, cl_mem y,
                                       const int offy);

template<typename Dtype>
void greentea_gpu_exp(const int ctx_id, const int N, const cl_mem a,
                      const int offa, cl_mem y, const int offy) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_exp = program.get_kernel(CL_KERNEL_SELECT("exp"));
  viennacl::ocl::enqueue(
      oclk_exp(N, WrapHandle(a, ctx), offa, WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_exp<float>(const int ctx_id, const int N,
                                      const cl_mem a, const int offa, cl_mem y,
                                      const int offy);
template void greentea_gpu_exp<double>(const int ctx_id, const int N,
                                       const cl_mem a, const int offa, cl_mem y,
                                       const int offy);

template<typename Dtype>
void greentea_gpu_powx(const int ctx_id, const int N, const cl_mem a,
                       const int offa, const Dtype alpha, cl_mem y,
                       const int offy) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_powx = program.get_kernel(
      CL_KERNEL_SELECT("powx"));
  viennacl::ocl::enqueue(
      oclk_powx(N, WrapHandle(a, ctx), offa, alpha, WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_powx<float>(const int ctx_id, const int N,
                                       const cl_mem a, const int offa,
                                       const float alpha, cl_mem y,
                                       const int offy);
template void greentea_gpu_powx<double>(const int ctx_id, const int N,
                                        const cl_mem a, const int offa,
                                        const double alpha, cl_mem y,
                                        const int offy);

template<typename Dtype>
void greentea_gpu_log(const int ctx_id, const int N, const cl_mem a,
                      const int offa, cl_mem y, const int offy) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_log = program.get_kernel(CL_KERNEL_SELECT("log"));
  viennacl::ocl::enqueue(
      oclk_log(N, WrapHandle(a, ctx), offa, WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_log<float>(const int ctx_id, const int N,
                                      const cl_mem a, const int offa, cl_mem y,
                                      const int offy);
template void greentea_gpu_log<double>(const int ctx_id, const int N,
                                       const cl_mem a, const int offa, cl_mem y,
                                       const int offy);

template<typename Dtype>
void greentea_gpu_sign(const int ctx_id, const int n, const cl_mem x, int offx,
                       cl_mem y, const int offy) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_sign = program.get_kernel(
      CL_KERNEL_SELECT("sign"));
  viennacl::ocl::enqueue(
      oclk_sign(n, WrapHandle(x, ctx), offx, WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_sign<float>(const int ctx_id, const int n,
                                       const cl_mem x, int offx, cl_mem y,
                                       const int offy);
template void greentea_gpu_sign<double>(const int ctx_id, const int n,
                                        const cl_mem x, int offx, cl_mem y,
                                        const int offy);

template<typename Dtype>
void greentea_gpu_sgnbit(const int ctx_id, const int n, const cl_mem x,
                         int offx, cl_mem y, const int offy) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(ctx_id);

  viennacl::ocl::kernel &oclk_sgnbit = program.get_kernel(
      CL_KERNEL_SELECT("sgnbit"));
  viennacl::ocl::enqueue(
      oclk_sgnbit(n, WrapHandle(x, ctx), offx, WrapHandle(y, ctx), offy),
      ctx.get_queue());
}

template void greentea_gpu_sgnbit<float>(const int ctx_id, const int n,
                                         const cl_mem x, int offx, cl_mem y,
                                         const int offy);
template void greentea_gpu_sgnbit<double>(const int ctx_id, const int n,
                                          const cl_mem x, int offx, cl_mem y,
                                          const int offy);

void greentea_gpu_rng_uniform(const int ctx_id, const int n, cl_mem r,
                              int offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<unsigned int> random(n);
  caffe_gpu_rng_uniform(n, &random[0]);
  greentea_gpu_memcpy(sizeof(unsigned int) * n, &random[0], r, offr, ctx);
}

template<typename Dtype>
void greentea_gpu_rng_uniform(const int ctx_id, const int n, const Dtype a,
                              const Dtype b, cl_mem r, const int offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<Dtype> random(n);
  caffe_rng_uniform(n, a, b, &random[0]);
  greentea_gpu_memcpy(sizeof(Dtype) * n, &random[0], r, offr, ctx);
}

template void greentea_gpu_rng_uniform<float>(const int ctx_id, const int n,
                                              const float a, const float b,
                                              cl_mem r, const int offr);
template void greentea_gpu_rng_uniform<double>(const int ctx_id, const int n,
                                               const double a, const double b,
                                               cl_mem r, const int offr);

template<typename Dtype>
void greentea_gpu_rng_gaussian(const int ctx_id, const int n, const Dtype mu,
                               const Dtype sigma, cl_mem r, const int offr) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);
  std::vector<Dtype> random(n);
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  greentea_gpu_memcpy(sizeof(Dtype) * n, &random[0], r, offr, ctx);
}

template void greentea_gpu_rng_gaussian<float>(const int ctx_id, const int n,
                                               const float mu,
                                               const float sigma, cl_mem r,
                                               const int offr);

template void greentea_gpu_rng_gaussian<double>(const int ctx_id, const int n,
                                                const double mu,
                                                const double sigma, cl_mem r,
                                                const int offr);

}
#endif
