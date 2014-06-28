// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/util/device.hpp"
#ifdef USE_OPENCL
#include "caffe/util/opencl_device.hpp"
#endif

namespace caffe {

template<typename Dtype>
Device<Dtype>* DeviceFactory<Dtype>::GetDevice() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return cpu_device_;
  case Caffe::GPU:
    return gpu_device_;
#ifdef USE_OPENCL
  case Caffe::OPENCL_CPU:
  case Caffe::OPENCL_GPU:
    return opencl_device_;
#endif
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<Device<Dtype>*>(NULL);
  }
}

template<typename Dtype>
Device<Dtype>* DeviceFactory<Dtype>::cpu_device_ = new CPUDevice<Dtype>();

template<typename Dtype>
Device<Dtype>* DeviceFactory<Dtype>::gpu_device_ = new GPUDevice<Dtype>();

#ifdef USE_OPENCL
template<typename Dtype>
Device<Dtype>* DeviceFactory<Dtype>::opencl_device_ = new OpenCLDevice<Dtype>();
#endif

INSTANTIATE_CLASS(DeviceFactory);

}  // namespace caffe
