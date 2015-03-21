#ifndef __OPENCL_DROPOUT_LAYER_HPP__
#define __OPENCL_DROPOUT_LAYER_HPP__
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

	template<typename T> bool clDropoutLayerForward(const int count, const T* bottom_data, const unsigned int* mask, const unsigned int threshold, T scale, T* top_data);
	template<typename T> bool clDropoutLayerBackward(const int count, const T* top_diff, const unsigned int* mask, const unsigned int threshold, T scale, T* bottom_diff);

} // namespace OpenCL

} // namespace caffee

#endif // __OPENCL_DROPOUT_LAYER_HPP__
