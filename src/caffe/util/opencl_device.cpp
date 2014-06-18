// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/util/opencl_device.hpp"

#include <vector>

namespace caffe {

template<typename Dtype>
cl_device_type OpenCLDevice<Dtype>::get_device_type() {
  switch (Caffe::mode()) {
  case Caffe::OPENCL_CPU:
    return CL_DEVICE_TYPE_CPU;
  case Caffe::OPENCL_GPU:
    return CL_DEVICE_TYPE_GPU;
  default:
    LOG(FATAL) << "Unknown Caffe OpenCL mode.";
    return CL_DEVICE_TYPE_DEFAULT;
  }
}

/**
 * http://dhruba.name/2012/08/14/opencl-cookbook-listing-all-devices-and-their-critical-attributes/
 */
template<typename Dtype>
cl_context OpenCLDevice<Dtype>::context() {
  if (cl_context_ == NULL) {
    cl_uint platformCount;
    CL_CHECK(clGetPlatformIDs(0, NULL, &platformCount));

    cl_platform_id* platforms = (cl_platform_id*)
        malloc(sizeof(cl_platform_id) * platformCount);
    CL_CHECK(clGetPlatformIDs(1, platforms, NULL));

    cl_uint deviceCount;
    cl_device_type device_type = get_device_type();
    int num_devices_to_skip = current_device_id_;
    while (num_devices_to_skip >= 0) {
      for (int i = 0; i < platformCount; i++) {
        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)(
                platforms[i]), 0};
        // get all devices
        clGetDeviceIDs(platforms[i], device_type, 0, NULL, &deviceCount);
        if (num_devices_to_skip <= deviceCount) {
          current_cl_platform_id_ = platforms[i];
          current_platform_device_count_ = deviceCount;
          current_platform_device_id_ = num_devices_to_skip;
          current_platform_device_ids_.resize(deviceCount);
          CL_CHECK(clGetDeviceIDs(current_cl_platform_id_, device_type,
                                  current_platform_device_count_,
                                  &(current_platform_device_ids_[0]), NULL));
          cl_int error = CL_SUCCESS;   // Used to handle error codes
          // TODO: clCreateContext or clCreateContextFromType?
  /*
   * http://dhruba.name/2012/10/14/opencl-cookbook-how-to-leverage-multiple-devices-in-opencl/
   */
  //        cl_context_ = clCreateContext(properties, deviceCount, devices, NULL,
  //                                    NULL, &error);
          cl_context_ = clCreateContextFromType(properties, device_type, NULL,
                                                NULL, &error);
          CL_CHECK(error);
        }
        num_devices_to_skip -= deviceCount;
        if (num_devices_to_skip < 0) {
          break;
        }
      }
    }
  }
  return cl_context_;
}

template<typename Dtype>
cl_device_id OpenCLDevice<Dtype>::current_cl_device_id() {
  // To initialize current platform info
  context();
  return current_platform_device_ids_[current_platform_device_id_];
}

template<typename Dtype>
cl_command_queue OpenCLDevice<Dtype>::queue() {
  if (cl_command_queue_ == NULL) {
    cl_int error = 0;   // Used to handle error codes
    cl_command_queue_properties properties = 0;
    cl_command_queue_ = clCreateCommandQueue(
        context(), current_cl_device_id(), properties, &error);
    CL_CHECK(error);
  }
  return cl_command_queue_;
}

template<typename Dtype>
void OpenCLDevice<Dtype>::release_context() {
  CL_CHECK(clReleaseContext(cl_context_));
  cl_context_ = NULL;
}

template<typename Dtype>
void OpenCLDevice<Dtype>::release_queue() {
  CL_CHECK(clReleaseCommandQueue(cl_command_queue_));
  cl_command_queue_ = NULL;
}

template<typename Dtype>
void OpenCLDevice<Dtype>::SetDevice(const int device_id) {
  if (current_device_id_ != device_id) {
    current_device_id_ = device_id;
    release_queue();
    // TODO: reuse context for the devices of the same platform
    release_context();
    context();
  }
}

template <>
void OpenCLDevice<float>::gemm(const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const int M,
                               const int N, const int K, const float alpha,
                               const float* A, const float* B,
                               const float beta, float* C) {
  int ldA = (TransA == CblasNoTrans) ? K : M;
  int ldB = (TransB == CblasNoTrans) ? N : K;
  int ldC = N;
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
      M, N, K, alpha, ARRAY(A), ARRAY(B), beta, ARRAY(C),
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
  int ldA = (TransA == CblasNoTrans) ? K : M;
  int ldB = (TransB == CblasNoTrans) ? N : K;
  int ldC = N;
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
      M, N, K, alpha, ARRAY(A), ARRAY(B), beta, ARRAY(C),
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
  int ldA = (TransA == CblasNoTrans) ? N : M;
  int ldx = N;
  int ldy = N;
  CREATE_CL_MEM(A, M, N, READ_ONLY);
  CREATE_CL_MEM(x, N, 1, READ_ONLY);
  CREATE_CL_MEM(y, M, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasSgemv(clblasRowMajor, clTransA, M, N, alpha,
      ARRAY(A), ARRAY(x), beta, ARRAY(y),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<double>::gemv(
    const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A,
    const double* x, const double beta, double* y) {
  clblasTranspose clTransA = to_clblasTranspose(TransA);
  int ldA = (TransA == CblasNoTrans) ? N : M;
  int ldx = N;
  int ldy = N;
  CREATE_CL_MEM(A, M, N, READ_ONLY);
  CREATE_CL_MEM(x, N, 1, READ_ONLY);
  CREATE_CL_MEM(y, M, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasDgemv(clblasRowMajor, clTransA, M, N, alpha,
      ARRAY(A), ARRAY(x), beta, ARRAY(y),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<float>::axpy(const int N, const float alpha,
                               const float* X, float* Y) {
  int ldX = N;
  int ldY = N;
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasSaxpy(
      N, alpha, ARRAY(X), ARRAY(Y),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<double>::axpy(const int N, const double alpha,
                                const double* X, double* Y) {
  int ldX = N;
  int ldY = N;
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasDaxpy(
      N, alpha, ARRAY(X), ARRAY(Y),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<float>::scal(const int N, const float alpha, float *X) {
  int ldX = N;
  CREATE_CL_MEM(X, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasSscal(
      N, alpha, ARRAY(X),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<double>::scal(const int N, const double alpha, double *X) {
  int ldX = N;
  CREATE_CL_MEM(X, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasDscal(
      N, alpha, ARRAY(X),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<float>::axpby(
    const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  this->scal(N, beta, Y);
  this->axpy(N, alpha, X, Y);
}

template <>
void OpenCLDevice<double>::axpby(
    const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  this->scal(N, beta, Y);
  this->axpy(N, alpha, X, Y);
}

template <>
void OpenCLDevice<float>::copy(const int N, const float *X, float *Y) {
  int ldX = N;
  int ldY = N;
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasScopy(
      N, ARRAY(X), ARRAY(Y),
      CLBALS_TRAILING_ARGS));
}

template <>
void OpenCLDevice<double>::copy(const int N, const double *X, double *Y) {
  int ldX = N;
  int ldY = N;
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
      OpenCLDevice::queue(), bufY, blocking_write, 0, N * sizeof(Dtype),
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
#ifdef CL_VERSION_1_2
  CREATE_CL_MEM(X, N, 1, READ_WRITE);
  cl_uint num_events_in_wait_list = 0;
  cl_event *event_wait_list = NULL;
  cl_event event = NULL;
  CL_CHECK(clEnqueueFillBuffer(
      OpenCLDevice::queue(), bufX, static_cast<void*>(&alpha), sizeof(Dtype),
      0, sizeof(Dtype) * N, num_events_in_wait_list, event_wait_list, &event));
#else
  std::vector<Dtype> tmp(N, alpha);
  copy_from_cpu(N, &tmp[0], X);
#endif
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

template
void OpenCLDevice<float>::add_scalar(const int N, const float alpha, float *X);
template
void OpenCLDevice<double>::add_scalar(const int N, const double alpha,
                                      double *X);


template<typename Dtype>
void OpenCLDevice<Dtype>::powx(const int N, const Dtype* a, const Dtype b,
                               Dtype* y) {
  NOT_IMPLEMENTED;
//  caffe_gpu_powx<Dtype>(N, a, b, y);
}

template
void OpenCLDevice<float>::powx(const int N, const float* a, const float b,
                               float *y);
template
void OpenCLDevice<double>::powx(const int N, const double* a,
                                const double b, double *y);


template<typename Dtype>
void OpenCLDevice<Dtype>::rng_uniform(const int N, const Dtype a,
                                      const Dtype b, Dtype* r) {
  NOT_IMPLEMENTED;
//  caffe_gpu_rng_uniform<Dtype>(N, a, b, r);
}

template
void OpenCLDevice<float>::rng_uniform(
    const int N, const float a, const float b, float* r);
template
void OpenCLDevice<double>::rng_uniform(
    const int N, const double a, const double b, double* r);

template<typename Dtype>
void OpenCLDevice<Dtype>::rng_gaussian(const int N, const Dtype mu,
                                       const Dtype sigma, Dtype* r) {
  NOT_IMPLEMENTED;
//  caffe_gpu_rng_gaussian<Dtype>(N, mu, sigma, r);
}

template
void OpenCLDevice<float>::rng_gaussian(
    const int N, const float mu, const float sigma, float* r);
template
void OpenCLDevice<double>::rng_gaussian(
    const int N, const double mu, const double sigma, double* r);

template<typename Dtype>
void OpenCLDevice<Dtype>::rng_bernoulli(const int N, const Dtype p, int* r) {
  NOT_IMPLEMENTED;
//  caffe_gpu_rng_bernoulli<Dtype>(N, p, r);
}

template
void OpenCLDevice<float>::rng_bernoulli(const int N, const float p, int* r);
template
void OpenCLDevice<double>::rng_bernoulli(const int N, const double p, int* r);

template<typename Dtype>
void OpenCLDevice<Dtype>::dot(const int N, const Dtype* x, const Dtype* y,
                              Dtype* out) {
  NOT_IMPLEMENTED;
//  caffe_gpu_dot<Dtype>(N, x, y, out);
}

template
void OpenCLDevice<float>::dot(const int N, const float* x, const float* y,
                              float* out);
template
void OpenCLDevice<double>::dot(const int N, const double* x, const double* y,
                               double* out);

template<typename Dtype>
void OpenCLDevice<Dtype>::hamming_distance(const int N, const Dtype* x,
                                           const Dtype* y, uint32_t* out) {
  NOT_IMPLEMENTED;
//  *out = caffe_gpu_hamming_distance<Dtype>(N, x, y);
}

template
void OpenCLDevice<float>::hamming_distance(const int N, const float* x,
                                           const float* y, uint32_t* out);
template
void OpenCLDevice<double>::hamming_distance(const int N, const double* x,
                                            const double* y, uint32_t* out);

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

template
void OpenCLDevice<float>::asum(const int N, const float* x, float* y);
template
void OpenCLDevice<double>::asum(const int N, const double* x, double* y);

template<typename Dtype>
void OpenCLDevice<Dtype>::scale(const int N, const Dtype alpha,
                                const Dtype *x, Dtype* y) {
  this->copy(N, x, y);
  this->scal(N, alpha, y);
}

template
void OpenCLDevice<float>::scale(const int N, const float alpha,
                                const float *x, float* y);
template
void OpenCLDevice<double>::scale(const int N, const double alpha,
                                 const double *x, double* y);

template<typename Dtype>
void OpenCLDevice<Dtype>::im2col(
    const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
//  NOT_IMPLEMENTED;
//  im2col_gpu(data_im, channels, height, width, ksize, pad, stride,
//             data_col);
}

template
void OpenCLDevice<float>::im2col(
    const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col);
template
void OpenCLDevice<double>::im2col(
    const double* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col);

template<typename Dtype>
void OpenCLDevice<Dtype>::col2im(
    const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im) {
//  NOT_IMPLEMENTED;
//  col2im_gpu(data_col, channels, height, width, psize, pad, stride,
//             data_im);
}

template
void OpenCLDevice<float>::col2im(
    const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im);
template
void OpenCLDevice<double>::col2im(
    const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im);

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

const char* clblasGetErrorString(clblasStatus status) {
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
