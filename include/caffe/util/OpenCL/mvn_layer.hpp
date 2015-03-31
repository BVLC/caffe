#ifndef __OPENCL_MVN_LAYER_HPP__
#define __OPENCL_MVN_LAYER_HPP__
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

	template<typename T> bool clMVNLayerForwardResidual(const T* bottom_data, const int bottom_data_height, const int bottom_data_width, const T* sum_multiplier, const int sum_multiplier_width, const T* mean, const int mean_width, const T* variance, const int variance_width, const T eps, T* top_data, const int top_data_height, const int top_data_width);

	template<typename T> bool clMVNLayerForwardMV2(const T* data2D, const int data2D_height, const int data2D_width, const T* data1D, const int data1D_length, T* linear_term, T* quadratic_term);
	template<typename T> bool clMVNLayerForward(const T* data2D_in, const int data2D_in_height, const int data2D_in_width, const T* data1D_in, const int data1D_in_length, const T* linear_term, const int linear_term_length, const T* quadratic_term, const int quadratic_term_length, const T eps, T* data2D_out);
	template<typename T> bool clMVNLayerForwardS2(const T* data2D_in, const int data2D_in_height, const int data2D_in_width, const T* data1D_in, const int data1D_in_length, T* data2D_out);

	template<typename T> bool clMVNLayerBackwardS1(const T* data2D_in, const T* diff2D_in, const int data2D_in_height, const int data2D_in_width, const T* data1D_in, const int data1D_in_length, const T* linear_term, const int linear_term_length,	const T* quadratic_term, const int quadratic_term_length, T* data2D_out);
	template<typename T> bool clMVNLayerBackwardMV2(const T* data2D, const T* diff2D, const int data2D_height, const int data2D_width, const T* data1D, const int data1D_length, T* linear_term, T* quadratic_term);
	template<typename T> bool clMVNLayerBackward(const T* data2D_in, const int data2D_in_height, const int data2D_in_width, const T* data1D_in, const int data1D_in_length, const T* linear_term, const int linear_term_length, const T* quadratic_term, const int quadratic_term_length, const T eps, T* data2D_out);

	template<typename T> bool clMVNLayerBackward_perf(const T* A2D_top, const T* A2D_top_diff, const int top_height, const int top_width, const T* A2D_bottom, const T* A2D_bottom_diff, const int bottom_height, const int bottom_width, const T* A1D_sum_multiplier, const T* A1D_buffer, const int sum_multiplier_length, const T eps, T* data2D_out);
	template<typename T> bool clMVNLayerForward_perf(const T* A2D_top, const T* A2D_top_diff, const int top_height, const int top_width, const T* A2D_bottom, const T* A2D_bottom_diff, const int bottom_height, const int bottom_width, const T* A1D_sum_multiplier, const T* A1D_buffer, const int sum_multiplier_length, const T eps, T* data2D_out);

} // namespace OpenCL

} // namespace caffee

#endif // __OPENCL_MVN_LAYER_HPP__
