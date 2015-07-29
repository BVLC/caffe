#ifndef __OPENCL_BNLL_LAYER_HPP__
#define __OPENCL_BNLL_LAYER_HPP__

#include <CL/cl.h>
#include <glog/logging.h>

#include <caffe/util/OpenCL/OpenCLDevice.hpp>
#include <caffe/util/OpenCL/OpenCLManager.hpp>
#include <caffe/util/OpenCL/OpenCLPlatform.hpp>
#include <caffe/util/OpenCL/OpenCLSupport.hpp>

#include <iostream>   // NOLINT(*)
#include <sstream>    // NOLINT(*)
#include <string>

namespace caffe {

namespace OpenCL {

template<typename T>
bool clSoftmaxLossForwardGPU(
    const int nthreads,
    const T* prob_data,
    const T* label,
    T* loss,
    const int num,
    const int dim,
    const int spatial_dim,
    const bool has_ignore_label_,
    const int ignore_label_,
    T* counts);
template<typename T>
bool clSoftmaxLossBackwardGPU(
    const int nthreads,
    const T* top,
    const T* label,
    T* bottom_diff,
    const int num,
    const int dim,
    const int spatial_dim,
    const bool has_ignore_label_,
    const int ignore_label_,
    T* counts);

}  // namespace OpenCL

}  // namespace caffe

#endif  // __OPENCL_BNLL_LAYER_HPP__
