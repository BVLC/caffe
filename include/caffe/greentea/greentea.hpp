/*
 * greentea.hpp
 *
 *  Created on: Apr 5, 2015
 *      Author: Fabian Tschopp
 */

#ifndef CAFFE_GREENTEA_HPP_
#define CAFFE_GREENTEA_HPP_

// Define ViennaCL/GreenTea flags
#ifdef USE_GREENTEA
#ifndef NDEBUG
#define NDEBUG
#endif

#ifndef VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_OPENCL
#endif

#include "CL/cl.h"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/backend/opencl.hpp"
#include "viennacl/vector.hpp"
#endif

namespace caffe {

#ifdef USE_GREENTEA
/*template<typename Dtype>
cl_mem Subregion(cl_mem in, size_t off, size_t size);*/

viennacl::ocl::handle<cl_mem> WrapHandle(cl_mem in, viennacl::ocl::context &ctx);
#endif

enum Backend {
  BACKEND_CUDA,
  BACKEND_OpenCL
};

class DeviceContext {
 public:
  DeviceContext();
  DeviceContext(int id, Backend backend);
  Backend backend() const;
  int id() const;
 private:
  int id_;
  Backend backend_;
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

#ifdef USE_VIENNACLBLAS
#define GREENTEA_VCL_BLAS_CHECK(condition) \
    ViennaCLStatus status = condition; \
    CHECK_EQ(status, ViennaCLSuccess) << "GreenTea ViennaCL BLAS ERROR";
#endif

#ifdef USE_CLBLAS
#define GREENTEA_CL_BLAS_CHECK(condition) \
    clblasStatus status = condition; \
    CHECK_EQ(status, clblasSuccess) << "GreenTea CL BLAS ERROR";
#endif

// Macro to select the single (_s) or double (_d) precision kernel
#define CL_KERNEL_SELECT(kernel) is_same<Dtype, float>::value ? kernel "_s" : kernel "_d"

#endif

}

#endif /* CAFFE_GREENTEA_HPP_ */
