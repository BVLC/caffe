#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#endif

#ifdef CPU_ONLY  // CPU-only Caffe.

#define CAFFE_CUDA_NUM_THREADS  0

#include <vector>

// Stub out GPU calls as unavailable.

#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#else  // Normal GPU + CPU Caffe.
#ifdef USE_CUDA  // Include CUDA macros and headers only if enabled

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types

//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int_tp i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace caffe {

// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

#define CAFFE_CUDA_NUM_THREADS 512

// CDT hacks: allow proper code formatting and remove errors in CDT
#ifdef __CDT_PARSER__
#include "device_launch_parameters.h"
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...)  <<< __VA_ARGS__ >>>
#endif

// CUDA: number of blocks for threads.
inline int_tp CAFFE_GET_BLOCKS(const int_tp N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

}  // namespace caffe


#endif  // USE_CUDA
#ifdef USE_GREENTEA
#define OCL_CHECK(condition) \
  do { \
    cl_int error = (condition); \
    CHECK_EQ(error, CL_SUCCESS) << " " << caffe::clGetErrorString(error); \
  } while (0)

#ifdef USE_FFT
#include "caffe/util/cl_fft_state.hpp"
#define CLFFT_CHECK(condition) \
  do { \
    clfftStatus status = (condition); \
    CHECK_EQ(status, CLFFT_SUCCESS) << " " \
      << caffe::clfftGetErrorString(status); \
  } while (0)

#endif  // USE_FFT

namespace caffe {

#ifdef USE_FFT
const char* clfftGetErrorString(clfftStatus status);
#endif

const char* clGetErrorString(cl_int error);

#define OCL_LOCAL_WORKGROUP_SIZE 256

// OCL: number of work groups
inline int CAFFE_GET_BLOCKS_OCL(const int N) {
  return (N + OCL_LOCAL_WORKGROUP_SIZE - 1) / OCL_LOCAL_WORKGROUP_SIZE;
}
inline int CAFFE_GET_BLOCKS_OCL(const int N, const int lws) {
  return (N + lws - 1) / lws;
}

// OCL: get padded global work size
inline int CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(const int N) {
  return CAFFE_GET_BLOCKS_OCL(N) * OCL_LOCAL_WORKGROUP_SIZE;
}
inline int CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(const int N, const int lws) {
  return CAFFE_GET_BLOCKS_OCL(N, lws) * lws;
}

}  // namespace caffe
#endif  // USE_GRREENTEA
#endif  // !CPU_ONLY

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
