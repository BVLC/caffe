#include "caffe/backend/device_kernel.hpp"

namespace caffe {

DeviceKernel::DeviceKernel(Device* dev, string name, KernelArgs args)
  : name_(name), device_(dev), arg_idx_(0),  args_(args) {
}

void DeviceKernel::add_arg(const bool *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const char *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const int8_t *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const int16_t *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const int32_t *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const int64_t *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const uint8_t *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const uint16_t *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const uint32_t *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const uint64_t *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const half_fp *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const float *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(const double *arg) {
  this->set_arg(arg_idx_, arg);
}

void DeviceKernel::add_arg(vptr<char> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<int8_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<int16_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<int32_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<int64_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<uint8_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<uint16_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<uint32_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<uint64_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<half_fp> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<float> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<double> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<void> *arg) {
  this->set_arg(arg_idx_, arg);
}

void DeviceKernel::add_arg(vptr<const char> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const int8_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const int16_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const int32_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const int64_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const uint8_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const uint16_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const uint32_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const uint64_t> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const half_fp> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const float> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const double> *arg) {
  this->set_arg(arg_idx_, arg);
}
void DeviceKernel::add_arg(vptr<const void> *arg) {
  this->set_arg(arg_idx_, arg);
}

}  // namespace caffe
