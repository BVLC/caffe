// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/device.hpp"
#include "caffe/devices/cpu.hpp"
#include "caffe/devices/gpu.hpp"

namespace caffe {


template<typename Dtype>
class DeviceFactory {
 public:
  static Device<Dtype>* GetDevice(Caffe::Brew mode);
 private:
  static Device<Dtype>* cpu_device_;
  static Device<Dtype>* gpu_device_;
};

template<typename Dtype>
Device<Dtype>* DeviceFactory<Dtype>::GetDevice(Caffe::Brew mode) {
  if (mode == Caffe::UNSPECIFIED)
    mode = Caffe::mode();
  switch (mode) {
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
template class DeviceFactory<int>;
template class DeviceFactory<unsigned int>;

// A function to get the device either for the current mode (if
// Caffe::UNSPECIFIED is passed, which is the default), or for a specific mode
// (if a specific device's Caffe::Brew is passed).
template<typename Dtype>
Device<Dtype>* GetDevice(Caffe::Brew mode) {
  return DeviceFactory<Dtype>::GetDevice(mode);
}

template Device<float>* GetDevice(Caffe::Brew mode);
template Device<double>* GetDevice(Caffe::Brew mode);
template Device<int>* GetDevice(Caffe::Brew mode);
template Device<unsigned int>* GetDevice(Caffe::Brew mode);

}  // namespace caffe
