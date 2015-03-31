#ifndef __OPENCL_THRESHOLD_LAYER_HPP__
#define __OPENCL_THRESHOLD_LAYER_HPP__
#include <sstream>
#include <iostream>
#include <string>
#include <CL/cl.h>

#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/OpenCLPlatform.hpp>
#include <caffe/util/OpenCL/OpenCLManager.hpp>
#include <caffe/util/OpenCL/OpenCLSupport.hpp>
#include <glog/logging.h>

namespace caffe {

namespace OpenCL {

	template<typename T> bool clThresholdForward(const int n, const T threshold, const T* in, T* out);
} // namespace OpenCL

} // namespace caffee

#endif // __OPENCL_THRESHOLD_LAYER_HPP__
