#ifndef CAFFE_BACKEND_OPENCL_OCL_MATH_HPP_
#define CAFFE_BACKEND_OPENCL_OCL_MATH_HPP_

#include "caffe/common.hpp"
#include "caffe/backend/device.hpp"

#ifdef USE_OPENCL
#include "ocl_device.hpp"

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/thread.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"

#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

#if defined(USE_CLBLAS)
#include <clBLAS.h>       // NOLINT
#endif  // USE_CLBLAS
#if defined(USE_CLBLAST)
#include <clblast.h>      // NOLINT
#endif  // USE_CLBLAST
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

namespace clblast {

#if defined(USE_CLBLAST)
#if defined(USE_HALF)
#include  <clblast_half.h>
#endif  // USE_HALF
#endif  // USE_CLBLAST

}


// ViennaCL 1.5.1 compatibility fix
#ifndef VIENNACL_MINOR_VERSION
#define VIENNACL_MINOR_VERSION 5
#endif

#if VIENNACL_MINOR_VERSION > 5
#define VCL_ROW_MAJOR , true
#define VCL_COL_MAJOR , false
#else
#define VCL_ROW_MAJOR
#define VCL_COL_MAJOR
#endif

#endif  // USE_OPENCL

#endif  // CAFFE_BACKEND_OPENCL_OCL_MATH_HPP_
