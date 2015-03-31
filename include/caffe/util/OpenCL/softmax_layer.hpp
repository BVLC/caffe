#ifndef __OPENCL_SOFTMAX_LAYER_HPP__
#define __OPENCL_SOFTMAX_LAYER_HPP__
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

	template<typename T> bool clkernel_channel_max(const int num, const int channels, const int spatial_dim, const T* data, T* out);
	template<typename T> bool clkernel_channel_subtract(const int num, const int channels, const int spatial_dim, T* data, const T* channel_max);
	template<typename T> bool clkernel_exp(const int count, const T* data, T* out);
	template<typename T> bool clkernel_channel_sum(const int num, const int channels, const int spatial_dim, const T* data, T* channel_sum);
	template<typename T> bool clkernel_channel_div(const int num, const int channels, const int spatial_dim, T* data, const T* channel_sum);
	template<typename T> bool clkernel_channel_dot(const int num, const int channels, const int spatial_dim, const T* data_1, const T* data_2, T* channel_dot);

} // namespace OpenCL

} // namespace caffee

#endif // __OPENCL_TANH_LAYER_HPP__
