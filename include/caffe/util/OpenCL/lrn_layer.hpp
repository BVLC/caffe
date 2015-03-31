#ifndef __OPENCL_LRN_LAYER_HPP__
#define __OPENCL_LRN_LAYER_HPP__
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

	template<typename T> bool clLRNFillScale(const int nthreads, const T* in, const int num, const int channels, const int height, const int width, const int size, const T alpha_over_size, T* scale);
	template<typename T> bool clLRNComputeOutput(const int nthreads, const T* in, const T* scale, const T negative_beta, T* out);
	template<typename T> bool clLRNComputeDiff(const int nthreads, const T* bottom_data, const T* top_data, const T* scale, const T* top_diff, const int num, const int channels, const int height, const int width, const int size, const T negative_beta, const T cache_ratio, T* bottom_diff);

} // namespace OpenCL

} // namespace caffee

#endif // __OPENCL_BNLL_LAYER_HPP__
