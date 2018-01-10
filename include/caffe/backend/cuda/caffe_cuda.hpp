#ifndef CAFFE_BACKEND_CUDA_CAFFE_CUDA_HPP_
#define CAFFE_BACKEND_CUDA_CAFFE_CUDA_HPP_

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#endif  // USE_CUDA


namespace caffe {

#ifdef USE_CUDA

//
// CUDA macros
//

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.

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
  for (int_tp i = blockIdx.X * blockDim.X + threadIdx.X; \
       i < (n); \
       i += blockDim.X * gridDim.X)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: library error reporting.
const char* cudaGetErrorString(CUresult error);
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CDT hacks: allow proper code formatting and remove errors in CDT
#ifdef __CDT_PARSER__
#include "device_launch_parameters.h"
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...)  <<< __VA_ARGS__ >>>
#endif


#endif  // USE_CUDA

}

#endif  // CAFFE_BACKEND_CUDA_CAFFE_CUDA_HPP_
