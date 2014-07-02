// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/device.hpp"

namespace caffe {

template<typename Dtype>
Device<Dtype>*
DeviceFactory<Dtype>::GetDevice() {
  switch (Caffe::mode()) {
    case Caffe::CPU:
      return cpu_device_;
    case Caffe::GPU:
      return gpu_device_;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
      return static_cast<Device<Dtype>*>(NULL);
  }
}

template<typename Dtype>
Device<Dtype>* DeviceFactory<Dtype>::cpu_device_ = new CPUDevice<Dtype>();

template<typename Dtype>
Device<Dtype>* DeviceFactory<Dtype>::gpu_device_ = new GPUDevice<Dtype>();

INSTANTIATE_CLASS(DeviceFactory);

}  // namespace caffe
