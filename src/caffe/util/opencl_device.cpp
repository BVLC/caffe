// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/util/opencl_device.hpp"

namespace caffe {

template<typename Dtype>
void OpenCLDevice<Dtype>::gemm(const CBLAS_TRANSPOSE TransA,
                                 const CBLAS_TRANSPOSE TransB, const int M,
                                 const int N, const int K, const Dtype alpha,
                                 const Dtype* A, const Dtype* B,
                                 const Dtype beta, Dtype* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = (TransA == CblasNoTrans) ? N : M;
  clblasTranspose clTransA = to_clblasTranspose(TransA);
  clblasTranspose clTransB = to_clblasTranspose(TransB);
  CREATE_CL_MEM(A, M, K, READ_ONLY);
  CREATE_CL_MEM(B, K, N, READ_ONLY);
  CREATE_CL_MEM(C, M, N, READ_WRITE);
  ENQUEUE_CL_BUFFER(Write, A, M, K);
  ENQUEUE_CL_BUFFER(Write, B, K, N);
  ENQUEUE_CL_BUFFER(Write, C, M, N);
  PRE_CLBLAS_CALL;
  // bufX is defined by the macro CREATE_CL_MEM(X, ...)
  CLBLAS_CHECK(clblasSgemm(clblasRowMajor, clTransA, clTransB,
      M, N, K, &alpha, bufA, 0, lda, bufB, 0, ldb, &beta, bufC, 0, ldc
      numCommandQueues, Caffe::opencl_queue(), numEventsInWaitList,
      eventWaitList, &events));
  /* Release OpenCL memory objects. */
  RELEASE_CL_MEM(C);
  RELEASE_CL_MEM(B);
  RELEASE_CL_MEM(A);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                                 const int N, const Dtype alpha, const Dtype* A,
                                 const Dtype* x, const Dtype beta, Dtype* y) {
  caffe_gpu_gemv<Dtype>(TransA, M, N, alpha, A, x, beta, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::axpy(const int N, const Dtype alpha, const Dtype* X,
                                 Dtype* Y) {
  caffe_gpu_axpy<Dtype>(N, alpha, X, Y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::axpby(const int N, const Dtype alpha,
                                  const Dtype* X, const Dtype beta, Dtype* Y) {
  caffe_gpu_axpby<Dtype>(N, alpha, X, beta, Y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::copy(const int N, const Dtype *X, Dtype *Y) {
  caffe_gpu_copy<Dtype>(N, X, Y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::copy_from_cpu(const int N, const Dtype *X, Dtype *Y) {
  CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyHostToDevice));
}

template<typename Dtype>
void OpenCLDevice<Dtype>::set(const int N, const Dtype alpha, Dtype *X) {
  caffe_gpu_set<Dtype>(N, alpha, X);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::add_scalar(const int N, const Dtype alpha,
                                       Dtype *X) {
  caffe_gpu_add_scalar<Dtype>(N, alpha, X);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::scal(const int N, const Dtype alpha, Dtype *X) {
  caffe_gpu_scal<Dtype>(N, alpha, X);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::sqr(const int N, const Dtype* a, Dtype* y) {
  NOT_IMPLEMENTED;
//  caffe_gpu_sqr<Dtype>(N, a, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::add(const int N, const Dtype* a, const Dtype* b,
                                Dtype* y) {
  caffe_gpu_add<Dtype>(N, a, b, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::sub(const int N, const Dtype* a, const Dtype* b,
                                Dtype* y) {
  caffe_gpu_sub<Dtype>(N, a, b, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::mul(const int N, const Dtype* a, const Dtype* b,
                                Dtype* y) {
  caffe_gpu_mul<Dtype>(N, a, b, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::div(const int N, const Dtype* a, const Dtype* b,
                                Dtype* y) {
  caffe_gpu_div<Dtype>(N, a, b, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::powx(const int N, const Dtype* a, const Dtype b,
                                 Dtype* y) {
  caffe_gpu_powx<Dtype>(N, a, b, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::rng_uniform(const int N, const Dtype a,
                                        const Dtype b, Dtype* r) {
  caffe_gpu_rng_uniform<Dtype>(N, a, b, r);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::rng_gaussian(const int N, const Dtype mu,
                                         const Dtype sigma, Dtype* r) {
  caffe_gpu_rng_gaussian<Dtype>(N, mu, sigma, r);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::rng_bernoulli(const int N, const Dtype p, int* r) {
  NOT_IMPLEMENTED;
//  caffe_gpu_rng_bernoulli<Dtype>(N, p, r);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::exp(const int N, const Dtype* a, Dtype* y) {
  NOT_IMPLEMENTED;
//  caffe_gpu_exp<Dtype>(N, a, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::dot(const int N, const Dtype* x, const Dtype* y,
                                Dtype* out) {
  caffe_gpu_dot<Dtype>(N, x, y, out);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::hamming_distance(const int N, const Dtype* x,
                                             const Dtype* y, uint32_t* out) {
  *out = caffe_gpu_hamming_distance<Dtype>(N, x, y);
}

template<typename Dtype>
// Returns the sum of the absolute values of the elements of vector x
void OpenCLDevice<Dtype>::asum(const int N, const Dtype* x, Dtype* y) {
  caffe_gpu_asum<Dtype>(N, x, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::sign(const int N, const Dtype* x, Dtype* y) {
  caffe_gpu_sign<Dtype>(N, x, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::sgnbit(const int N, const Dtype* x, Dtype* y) {
  caffe_gpu_sgnbit<Dtype>(N, x, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::fabs(const int N, const Dtype* x, Dtype* y) {
  caffe_gpu_fabs<Dtype>(N, x, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::scale(const int N, const Dtype alpha,
                                  const Dtype *x, Dtype* y) {
  caffe_gpu_scale<Dtype>(N, alpha, x, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::im2col(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  im2col_gpu(data_im, channels, height, width, ksize, pad, stride,
             data_col);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::col2im(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im) {
  col2im_gpu(data_col, channels, height, width, psize, pad, stride,
             data_im);
}

INSTANTIATE_CLASS(OpenCLDevice);

const char* clGetErrorString(cl_int error) {
  switch (error) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
  case CL_INVALID_COMMAND_QUEUE:
    return "CL_INVALID_COMMAND_QUEUE";
  case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";
  case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";
  case CL_INVALID_EVENT_WAIT_LIST:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  }
  return "Unknown OpenCL error";
}

const char* clblasGetErrorString(clblasStatus_t status) {
  switch (status) {
  case clblasSuccess:
    return "clblasSuccess";
  case clblasInvalidValue:
    return "clblasInvalidValue";
  case clblasInvalidCommandQueue:
    return "clblasInvalidCommandQueue";
  case clblasInvalidContext:
    return "clblasInvalidContext";
  case clblasInvalidMemObject:
    return "clblasInvalidMemObject";
  case clblasInvalidDevice:
    return "clblasInvalidDevice";
  case clblasInvalidEventWaitList:
    return "clblasInvalidEventWaitList";
  case clblasOutOfResources:
    return "clblasOutOfResources";
  case clblasOutOfHostMemory:
    return "clblasOutOfHostMemory";
  case clblasInvalidOperation:
    return "clblasInvalidOperation";
  case clblasCompilerNotAvailable:
    return "clblasCompilerNotAvailable";
  case clblasBuildProgramFailure:
    return "clblasBuildProgramFailure";
  case clblasNotImplemented:
    return "clblasNotImplemented";
  case clblasNotInitialized:
    return "clblasNotInitialized";
  case clblasInvalidMatA:
    return "clblasInvalidMatA";
  case clblasInvalidMatB:
    return "clblasInvalidMatB";
  case clblasInvalidMatC:
    return "clblasInvalidMatC";
  case clblasInvalidVecX:
    return "clblasInvalidVecX";
  case clblasInvalidVecY:
    return "clblasInvalidVecY";
  case clblasInvalidDim:
    return "clblasInvalidDim";
  case clblasInvalidLeadDimA:
    return "clblasInvalidLeadDimA";
  case clblasInvalidLeadDimB:
    return "clblasInvalidLeadDimB";
  case clblasInvalidLeadDimC:
    return "clblasInvalidLeadDimC";
  case clblasInvalidIncX:
    return "clblasInvalidIncX";
  case clblasInvalidIncY:
    return "clblasInvalidIncY";
  case clblasInsufficientMemMatA:
    return "clblasInsufficientMemMatA";
  case clblasInsufficientMemMatB:
    return "clblasInsufficientMemMatB";
  case clblasInsufficientMemMatC:
    return "clblasInsufficientMemMatC";
  case clblasInsufficientMemVecX:
    return "clblasInsufficientMemVecX";
  case clblasInsufficientMemVecY:
    return "clblasInsufficientMemVecY";
  }
  return "Unknown clblas status";
}

}  // namespace caffe
