// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/util/opencl_device.hpp"

namespace caffe {

DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(sqr, y[i] = x[i] * x[i]);
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(exp, y[i] = exp(x[i]));
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(sign, y[i] = sign<Dtype>(x[i]));
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(sgnbit, y[i] = signbit(x[i]));
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(fabs, y[i] = fabs(x[i]));

DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(add, y[i] = a[i] + b[i]);
DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(sub, y[i] = a[i] - b[i]);
DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(mul, y[i] = a[i] * b[i]);
DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(div, y[i] = a[i] / b[i]);

template <>
void OpenCLDevice<float>::gemm(const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const int M,
                               const int N, const int K, const float alpha,
                               const float* A, const float* B,
                               const float beta, float* C) {
  // Note that cublas follows fortran order.
  LEAD_DIM(A, M, K);
  LEAD_DIM(B, K, N);
  LEAD_DIM(C, M, N);
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
      M, N, K, &alpha, ARRAY(A), ARRAY(B), &beta, ARRAY(C),
      CLBALS_TRAILING_ARGS));
  /* Release OpenCL memory objects. */
  RELEASE_CL_MEM(C);
  RELEASE_CL_MEM(B);
  RELEASE_CL_MEM(A);
}

template <>
void OpenCLDevice<double>::gemm(const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB, const int M,
                                const int N, const int K, const double alpha,
                                const double* A, const double* B,
                                const double beta, double* C) {
  // Note that cublas follows fortran order.
  LEAD_DIM(A, M, K);
  LEAD_DIM(B, K, N);
  LEAD_DIM(C, M, N);
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
  CLBLAS_CHECK(clblasDgemm(clblasRowMajor, clTransA, clTransB,
      M, N, K, &alpha, ARRAY(A), ARRAY(B), &beta, ARRAY(C),
      CLBALS_TRAILING_ARGS));
  /* Release OpenCL memory objects. */
  RELEASE_CL_MEM(C);
  RELEASE_CL_MEM(B);
  RELEASE_CL_MEM(A);
}

template <>
void OpenCLDevice<float>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
                               const int N, const float alpha, const float* A,
                               const float* x, const float beta, float* y) {
  clblasTranspose clTransA = to_clblasTranspose(TransA);
  CREATE_CL_MEM(A, M, N, READ_ONLY);
  CREATE_CL_MEM(x, N, 1, READ_ONLY);
  CREATE_CL_MEM(y, M, 1, READ_WRITE);
  CLBLAS_CHECK(clblasSgemv(clblasRowMajor, clTransA, M, N, &alpha,
      ARRAY(A), ARRAY(x), &beta, ARRAY(y),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<double>::gemv(
    const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A,
    const double* x, const double beta, double* y) {
  clblasTranspose clTransA = to_clblasTranspose(TransA);
  CREATE_CL_MEM(A, M, N, READ_ONLY);
  CREATE_CL_MEM(x, N, 1, READ_ONLY);
  CREATE_CL_MEM(y, M, 1, READ_WRITE);
  CLBLAS_CHECK(clblasDgemv(clblasRowMajor, clTransA, M, N, &alpha,
      ARRAY(A), ARRAY(x), &beta, ARRAY(y),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<float>::axpy(const int N, const float alpha,
                               const float* X, float* Y) {
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CUBLAS_CHECK(clblasSaxpy(
      N, &alpha, ARRAY(X), ARRAY(Y),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<double>::axpy(const int N, const double alpha,
                                const double* X, double* Y) {
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CUBLAS_CHECK(clblasDaxpy(
      N, &alpha, ARRAY(X), ARRAY(Y),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<float>::axpby(
    const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  this->scal<float>(N, beta, Y);
  this->axpy<float>(N, alpha, X, Y);
}

template <>
void OpenCLDevice<double>::axpby(
    const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  this->scal<double>(N, beta, Y);
  this->axpy<double>(N, alpha, X, Y);
}

template <>
void OpenCLDevice<float>::copy(const int N, const float *X, float *Y) {
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasScopy(
      N, ARRAY(X), ARRAY(Y),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<double>::copy(const int N, const double *X, double *Y) {
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasDcopy(
      N, ARRAY(X), ARRAY(Y),
      CLBALS_TRAILING_ARGS));
}

/**
 * http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueWriteBuffer.html
 */
template<typename Dtype>
void OpenCLDevice<Dtype>::copy_from_cpu(const int N, const Dtype *X,
                                        Dtype *Y) {
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  cl_bool blocking_write = CL_TRUE;
  cl_uint num_events_in_wait_list = 0;
  cl_event *event_wait_list = NULL;
  cl_event events = NULL;
  CL_CHECK(clEnqueueWriteBuffer(
      Caffe::opencl_queue(), bufY, blocking_write, 0, N * sizeof(Dtype),
      X, num_events_in_wait_list, event_wait_list, &events));
}

template
void OpenCLDevice<float>::copy_from_cpu(const int N, const float *X,
                                        float *Y);
template
void OpenCLDevice<double>::copy_from_cpu(const int N, const double *X,
                                         double *Y);

/**
 * http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueFillBuffer.html
 */
template<typename Dtype>
void OpenCLDevice<Dtype>::set(const int N, const Dtype alpha, Dtype *X) {
  CREATE_CL_MEM(X, N, 1, READ_WRITE);
  cl_uint num_events_in_wait_list = 0;
  cl_event *event_wait_list = NULL;
  cl_event events = NULL;
  CL_CHECK(clEnqueueFillBuffer(
      Caffe::opencl_queue(), bufA, &alpha, sizeof(Dtype), 0,
      sizeof(Dtype) * N, num_events_in_wait_list, event_wait_list, &event));
}

template
void OpenCLDevice<float>::set(const int N, const float alpha, float *X);
template
void OpenCLDevice<double>::set(const int N, const double alpha, double *X);

template<typename Dtype>
void OpenCLDevice<Dtype>::add_scalar(const int N, const Dtype alpha,
                                     Dtype *X) {
  NOT_IMPLEMENTED;
}

template <>
void OpenCLDevice<float>::scal(const int N, const float alpha, float *X) {
  CREATE_CL_MEM(X, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasSscal(
      N, alpha, ARRAY(X),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<double>::scal(const int N, const double alpha, double *X) {
  CREATE_CL_MEM(X, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasDscal(
      N, alpha, ARRAY(X),
      CLBALS_TRAILING_ARGS));
}

template<typename Dtype>
void OpenCLDevice<Dtype>::sqr(const int N, const Dtype* a, Dtype* y) {
  NOT_IMPLEMENTED;
//  caffe_gpu_sqr<Dtype>(N, a, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::add(const int N, const Dtype* a, const Dtype* b,
                              Dtype* y) {
  caffe_opencl_add<Dtype>(N, a, b, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::sub(const int N, const Dtype* a, const Dtype* b,
                              Dtype* y) {
  caffe_opencl_sub<Dtype>(N, a, b, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::mul(const int N, const Dtype* a, const Dtype* b,
                              Dtype* y) {
  caffe_opencl_mul<Dtype>(N, a, b, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::div(const int N, const Dtype* a, const Dtype* b,
                              Dtype* y) {
  caffe_opencl_div<Dtype>(N, a, b, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::powx(const int N, const Dtype* a, const Dtype b,
                               Dtype* y) {
  NOT_IMPLEMENTED;
//  caffe_gpu_powx<Dtype>(N, a, b, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::rng_uniform(const int N, const Dtype a,
                                      const Dtype b, Dtype* r) {
  NOT_IMPLEMENTED;
//  caffe_gpu_rng_uniform<Dtype>(N, a, b, r);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::rng_gaussian(const int N, const Dtype mu,
                                       const Dtype sigma, Dtype* r) {
  NOT_IMPLEMENTED;
//  caffe_gpu_rng_gaussian<Dtype>(N, mu, sigma, r);
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
  NOT_IMPLEMENTED;
//  caffe_gpu_dot<Dtype>(N, x, y, out);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::hamming_distance(const int N, const Dtype* x,
                                           const Dtype* y, uint32_t* out) {
  NOT_IMPLEMENTED;
//  *out = caffe_gpu_hamming_distance<Dtype>(N, x, y);
}

/**
 *
clblasSasum(
    size_t N,
    cl_mem asum,
    size_t offAsum,
    const cl_mem X,
    size_t offx,
    int incx,
    cl_mem scratchBuff,
    cl_uint numCommandQueues,
    cl_command_queue *commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event *eventWaitList,
    cl_event *events)
 */
template<typename Dtype>
void OpenCLDevice<Dtype>::asum(const int N, const Dtype* x, Dtype* y) {
  NOT_IMPLEMENTED;
//  CREATE_CL_MEM(x, N, 1, READ_ONLY);
//  CREATE_CL_MEM(y, N, 1, READ_WRITE);
//  PRE_CLBLAS_CALL;
//  CLBLAS_CHECK(clblasSasum(
//      N, alpha, ARRAY(X),
//      CLBALS_TRAILING_ARGS));
}

template<typename Dtype>
void OpenCLDevice<Dtype>::sign(const int N, const Dtype* x, Dtype* y) {
  NOT_IMPLEMENTED;
//  caffe_gpu_sign<Dtype>(N, x, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::sgnbit(const int N, const Dtype* x, Dtype* y) {
  NOT_IMPLEMENTED;
//  caffe_gpu_sgnbit<Dtype>(N, x, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::fabs(const int N, const Dtype* x, Dtype* y) {
  NOT_IMPLEMENTED;
//  caffe_gpu_fabs<Dtype>(N, x, y);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::scale(const int N, const Dtype alpha,
                                const Dtype *x, Dtype* y) {
  this->copy<Dtype>(N, x, y);
  this->scal<Dtype>(N, alpha, y);
}

template
void OpenCLDevice<float>::scale(const int N, const float alpha,
                                const float *x, float* y);
template
void OpenCLDevice<double>::scale(const int N, const double alpha,
                                 const double *x, double* y);

template<typename Dtype>
void OpenCLDevice<Dtype>::im2col(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  NOT_IMPLEMENTED;
//  im2col_gpu(data_im, channels, height, width, ksize, pad, stride,
//             data_col);
}

template<typename Dtype>
void OpenCLDevice<Dtype>::col2im(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im) {
  NOT_IMPLEMENTED;
//  col2im_gpu(data_col, channels, height, width, psize, pad, stride,
//             data_im);
}

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
