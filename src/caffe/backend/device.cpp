#include <algorithm>
#include <string>
#include <vector>

#include "caffe/backend/device.hpp"

namespace caffe {

device::device()
    : current_queue_id_(0), workgroup_sizes_(3, 0), id_(0), list_id_(0),
      backend_(Backend::BACKEND_CPU), memory_usage_(0), peak_memory_usage_(0),
      host_unified_(false), name_("") {
}


Backend device::backend() const {
  return backend_;
}

uint_tp device::id() const {
  return id_;
}

uint_tp device::list_id() const {
  return list_id_;
}

uint_tp device::workgroup_size(uint_tp id) {
  return workgroup_sizes_[id % 3];
}


template<>
std::shared_ptr<Blob<half_float::half, half_float::half> > device::Buffer(uint_tp id) {
  if (buff_h_.size() <= id) {
    std::shared_ptr<Blob<half_float::half, half_float::half> > blob_pointer(new Blob<half_float:half>(this));
    buff_h_.push_back(blob_pointer);
  }
  return buff_h_[id];
}

template<>
std::shared_ptr<Blob<float, float> > device::Buffer(uint_tp id) {
  if (buff_f_.size() <= id) {
    std::shared_ptr<Blob<float, float> > blob_pointer(new Blob<float>(this));
    buff_f_.push_back(blob_pointer);
  }
  return buff_f_[id];
}

template<>
std::shared_ptr<Blob<double, double> > device::Buffer(uint_tp id) {
  if (buff_d_.size() <= id) {
    std::shared_ptr<Blob<double, double> > blob_pointer(new Blob<double>(this));
    buff_d_.push_back(blob_pointer);
  }
  return buff_d_[id];
}

uint_tp device::current_queue_id() {
  return current_queue_id_;
}

uint_tp device::memory_usage() {
  return memory_usage_;
}

uint_tp device::peak_memory_usage() {
  return peak_memory_usage_;
}

void device::IncreaseMemoryUsage(uint_tp bytes) {
  memory_usage_ += bytes;
  if (memory_usage_ > peak_memory_usage_) {
    peak_memory_usage_ = memory_usage_;
  }
}

void device::DecreaseMemoryUsage(uint_tp bytes) {
  memory_usage_ -= bytes;
}

void device::ResetPeakMemoryUsage() {
  peak_memory_usage_ = memory_usage_;
}


}  // namespace caffe
