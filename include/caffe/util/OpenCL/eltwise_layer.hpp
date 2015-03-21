#ifndef __OPENCL_ELTWISE_LAYER_HPP__
#define __OPENCL_ELTWISE_LAYER_HPP__
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

	template<typename T> bool clMaxForward(const int nthreads, const T* bottom_data_a, const T* bottom_data_b, const int blob_idx, T* top_data, int* mask);
	template<typename T> bool clMaxBackward(const int nthreads, const T* top_diff, const int blob_idx, const int* mask, T* bottom_diff);

} // namespace OpenCL

} // namespace caffee

#endif // __OPENCL_ELTWISE_LAYER_HPP__
