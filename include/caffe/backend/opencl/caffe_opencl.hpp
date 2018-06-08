#ifndef CAFFE_BACKEND_OPENCL_CAFFE_OPENCL_HPP_
#define CAFFE_BACKEND_OPENCL_CAFFE_OPENCL_HPP_

#ifdef CMAKE_BUILD
#include "caffe_config.h"
#endif

#include <vector>

#ifdef USE_OPENCL

#ifndef VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_OPENCL
#endif  // VIENNACL_PROFILING_ENABLED

#ifndef VIENNACL_PROFILING_ENABLED
#define VIENNACL_PROFILING_ENABLED
#endif  // VIENNACL_PROFILING_ENABLED

#ifndef __APPLE__
#include "CL/cl.h"
#else
#include "OpenCL/cl.h"
#endif


#if defined(USE_CLBLAS)
  #include <clBLAS.h>       // NOLINT
#endif
#if defined(USE_CLBLAST)
  #include <clblast.h>      // NOLINT
#endif
#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/vector.hpp"

namespace caffe {

#ifndef OPENCL_QUEUE_COUNT
#define OPENCL_QUEUE_COUNT 8
#endif  // OPENCL_QUEUE_COUNT

#if defined(USE_CLBLAS) && defined(USE_CLBLAST)
#error Only one of USE_CLBLAS and USE_CLBLAST can be defined!
#endif

#if defined(USE_CLBLAS)
#define OPENCL_CL_BLAS_CHECK(condition) \
    {clblasStatus status = condition; \
    CHECK_EQ(status, clblasSuccess) << \
    "OPENCL ERROR: clBLAS error";}
#endif

#if defined (USE_CLBLAST)
#define OPENCL_CLBLAST_CHECK(condition) \
    {clblast::StatusCode status = condition; \
    CHECK_EQ(\
      static_cast<int>(status), \
      static_cast<int>(clblast::StatusCode::kSuccess)) << \
    "OPENCL ERROR: CLBlast error";}
#endif

const char* clGetErrorString(cl_int error);

#define OCL_CHECK(condition) \
  do { \
    cl_int error = (condition); \
    CHECK_EQ(error, CL_SUCCESS) << " " << clGetErrorString(error); \
  } while (0)

#define OCL_CHECK_MESSAGE(condition, message) \
  do { \
    cl_int error = (condition); \
    CHECK_EQ(error, CL_SUCCESS) << " " << clGetErrorString(error) \
                                       << " (" << message << ")"; \
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


/*
#define OCL_LOCAL_WORKGROUP_SIZE 256

// OCL: number of work groups
inline int CAFFE_GET_BLOCKS_OCL(const int n) {
  return (n + OCL_LOCAL_WORKGROUP_SIZE - 1) / OCL_LOCAL_WORKGROUP_SIZE;
}
inline int CAFFE_GET_BLOCKS_OCL(const int n, const int lws) {
  return (n + lws - 1) / lws;
}

// OCL: get padded global work size
inline int CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(const int n) {
  return CAFFE_GET_BLOCKS_OCL(n) * OCL_LOCAL_WORKGROUP_SIZE;
}
inline int CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(const int n, const int lws) {
  return CAFFE_GET_BLOCKS_OCL(n, lws) * lws;
}
*/

}  // namespace caffe

#endif  // USE_OPENCL

#endif  // CAFFE_BACKEND_OPENCL_CAFFE_OPENCL_HPP_
