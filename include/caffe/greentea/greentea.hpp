/*
 * greentea.hpp
 *
 *  Created on: Apr 5, 2015
 *      Author: Fabian Tschopp
 */

#ifndef CAFFE_GREENTEA_HPP_
#define CAFFE_GREENTEA_HPP_

#define VIENNACL_PROFILING_ENABLED

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#endif

#include <vector>

// Define ViennaCL/GreenTea flags
#ifdef USE_GREENTEA
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

#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/vector.hpp"
#endif

#ifndef GREENTEA_QUEUE_COUNT
#define GREENTEA_QUEUE_COUNT 1
#endif

namespace caffe {

#ifdef USE_GREENTEA
viennacl::ocl::handle<cl_mem> WrapHandle(cl_mem in,
                                         viennacl::ocl::context *ctx);
#endif

enum Backend {
  BACKEND_CUDA,
  BACKEND_OpenCL,
  BACKEND_CPU
};


template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

#ifdef USE_GREENTEA

#if defined(USE_CLBLAS) && defined(USE_CLBLAST)
#error Only one of USE_CLBLAS and USE_CLBLAST can be defined!
#endif

#if defined (USE_CLBLAS)
#define GREENTEA_CL_BLAS_CHECK(condition) \
    {clblasStatus status = condition; \
    CHECK_EQ(status, clblasSuccess) << \
    "GREENTEA ERROR: clBLAS error";}
#endif

#if defined (USE_CLBLAST)
#define GREENTEA_CLBLAST_CHECK(condition) \
    {clblast::StatusCode status = condition; \
    CHECK_EQ(\
      static_cast<int>(status), \
      static_cast<int>(clblast::StatusCode::kSuccess)) << \
    "GREENTEA ERROR: CLBlast error";}
#endif

// Macro to select the single (_float) or double (_double) precision kernel
#define CL_KERNEL_SELECT(kernel) \
  is_same<Dtype, float>::value ? \
      kernel "_float" : \
      kernel "_double"

#endif

}  // namespace caffe

#endif  /* CAFFE_GREENTEA_HPP_ */
