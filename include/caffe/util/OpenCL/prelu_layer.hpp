#ifndef __OPENCL_PRELU_LAYER_HPP__
#define __OPENCL_PRELU_LAYER_HPP__
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

	template<typename T> bool clPReLUForward(const int n, const int channels, const int dim, const T* in, T* out, const T* slope_data, const int div_factor);
	template<typename T> bool clPReLUBackward(const int n, const int channels, const int dim, const T* in_diff, const T* in_data, T* out_diff, const T* slope_data, const int div_factor);
	template<typename T> bool clPReLUParamBackward(const int n, const T* in_diff, const T* in_data, T* out_diff);

} // namespace OpenCL

} // namespace caffee

#endif // __OPENCL_PRELU_LAYER_HPP__





