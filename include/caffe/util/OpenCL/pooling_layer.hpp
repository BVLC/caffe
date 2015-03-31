#ifndef __OPENCL_POOLING_LAYER_HPP__
#define __OPENCL_POOLING_LAYER_HPP__
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


	template<typename T> bool clMaxPoolBackward(const int nthreads, const T* top_diff, const int* mask, const T* top_mask, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, T* bottom_diff);
	template<typename T> bool clAvePoolBackward(const int nthreads, const T* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, T* bottom_diff);
	template<typename T> bool clStoPoolBackward(const int nthreads, const T* rand_idx, const T* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, T* bottom_diff);

	template<typename T> bool clMaxPoolForward(const int nthreads, const T* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, T* top_data, int* mask, T* top_mask);
	template<typename T> bool clAvePoolForward(const int nthreads, const T* botton_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, T* top_data, int* mask, T* top_mask);
	template<typename T> bool clStoPoolForwardTrain(const int nthreads, const T* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, T* rand_idx, T* top_data);
	template<typename T> bool clStoPoolForwardTest(const int nthreads, const T* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, T* top_data);

} // namespace OpenCL

} // namespace caffee

#endif // __OPENCL_POOLING_LAYER_HPP__





