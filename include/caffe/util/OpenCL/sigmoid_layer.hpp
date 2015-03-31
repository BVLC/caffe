#ifndef __OPENCL_SIGMOID_LAYER_HPP__
#define __OPENCL_SIGMOID_LAYER_HPP__
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

	template<typename T> bool clSigmoidLayerForward(const int count, const T* bottom_data, T* top_data);
	template<typename T> bool clSigmoidLayerBackward(const int count, const T* top_diff, const T* top_data, T* bottom_diff);
} // namespace OpenCL

} // namespace caffee

#endif // __OPENCL_SIGMOID_LAYER_HPP__
