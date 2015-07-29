#ifndef __OPENCL_RELU_LAYER_HPP__
#define __OPENCL_RELU_LAYER_HPP__

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

template<typename T> bool clReLULayerForward(
    const int count,
    const T* bottom_data,
    T* top_data,
    T negative_slope);
template<typename T> bool clReLULayerBackward(
    const int count,
    const T* top_diff,
    const T* bottom_data,
    T* bottom_diff,
    T negative_slope);
}  // namespace OpenCL

}  // namespace caffe

#endif  // __OPENCL_RELU_LAYER_HPP__

