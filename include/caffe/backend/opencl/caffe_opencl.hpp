#ifndef CAFFE_BACKEND_OPENCL_CAFFE_OPENCL_HPP_
#define CAFFE_BACKEND_OPENCL_CAFFE_OPENCL_HPP_

#define VIENNACL_PROFILING_ENABLED

#ifdef CMAKE_BUILD
#include "caffe_config.h"
#endif

#include <vector>

namespace caffe {

// Define ViennaCL/GreenTea flags
#ifdef USE_OPENCL
#ifndef NDEBUG
#define NDEBUG
#endif

#ifndef VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_OPENCL
#endif

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

#ifndef GREENTEA_QUEUE_COUNT
#define GREENTEA_QUEUE_COUNT 1
#endif

viennacl::ocl::handle<cl_mem> WrapHandle(cl_mem in,
                                         viennacl::ocl::context *ctx);
bool IsBeignet(viennacl::ocl::context *ctx);

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

#if defined(USE_CLBLAS) && defined(USE_CLBLAST)
#error Only one of USE_CLBLAS and USE_CLBLAST can be defined!
#endif

#if defined(USE_CLBLAS)
#define OPENCL_CL_BLAS_CHECK(condition) \
    {clblasStatus status = condition; \
    CHECK_EQ(status, clblasSuccess) << \
    "GREENTEA ERROR: clBLAS error";}
#endif

#if defined (USE_CLBLAST)
#define OPENCL_CLBLAST_CHECK(condition) \
    {clblast::StatusCode status = condition; \
    CHECK_EQ(\
      static_cast<int>(status), \
      static_cast<int>(clblast::StatusCode::kSuccess)) << \
    "GREENTEA ERROR: CLBlast error";}
#endif

// Macro to select the single (_float) or double (_double) precision kernel
#define CL_KERNEL_SELECT(kernel) \
  is_same<Dtype, float>::value ? \
      kernel "_float" : (is_same<Dtype, double>::value ?\
       kernel "_double" : kernel "_half")


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

#endif  // USE_OPENCL

}  // namespace caffe

#endif  // CAFFE_BACKEND_OPENCL_CAFFE_OPENCL_HPP_
